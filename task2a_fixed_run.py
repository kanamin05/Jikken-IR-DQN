#!/usr/bin/env python3
"""Fixed map-axis waypoints from a stopped Task2A pose. Preview by default.

No DQN inference, A*, move_base goal, controller switching or joint commands.
First drive forward at the current heading until goal Y, then strafe to goal X.
Preview never publishes velocity. Gazebo only.
"""
import argparse
import json
import math
import os
import time
import numpy as np
from scipy.spatial import cKDTree
from task2a_precise_run import SurfaceDQN
from task2a_cloud_step import create_session
from pointcloud_obstacles import rotation_matrix


def forward_waypoints(position, yaw, goal):
    """TASK2A demo change: no sideways entry into the narrow passage."""
    direction = np.array([math.cos(yaw), math.sin(yaw)])
    if abs(direction[1]) < .25:
        raise ValueError('Current heading is nearly parallel to goal Y; forward route refused')
    distance = (goal[1]-position[1])/direction[1]
    if distance < 0 or distance > 3:
        raise ValueError('Goal Y must be ahead and within 3 m')
    return [np.asarray(position)+distance*direction, np.asarray(goal)]


def segment_report(policy, start, target):
    points = np.linspace(start, target,
        max(2, int(np.ceil(np.linalg.norm(np.asarray(target)-start)/.001))+1))
    distances = policy.tree.query(points)[0]
    reasons = policy.segment_rejections(start, target)
    return dict(start=np.asarray(start).tolist(), target=np.asarray(target).tolist(),
                minimum_surface_clearance=float(distances.min()),
                end_surface_clearance=float(distances[-1]), rejected_by=reasons)


def main():
    import rospy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Bool, Float32, String
    from geometry_msgs.msg import Twist
    from actionlib_msgs.msg import GoalStatusArray
    rospy.init_node('task2a_fixed_run')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--goal-x', type=float, default=2.)
    parser.add_argument('--goal-y', type=float, default=3.5)
    parser.add_argument('--seed-run', help='Use the stopped surface memory from this same scene/run')
    parser.add_argument('--weights', default=os.path.join(os.path.dirname(__file__), 'task2a_policy_numpy.npz'))
    parser.add_argument('--base-frame', default='base_footprint')
    parser.add_argument('--diagnostics-dir', default='diagnostics/fixed_run')
    args = parser.parse_args(rospy.myargv()[1:])
    if not np.isfinite([args.goal_x, args.goal_y]).all():
        parser.error('Waypoint coordinates must be finite')
    if not rospy.get_param('/use_sim_time', False):
        parser.error('This runner is for the calibrated Gazebo scene')
    if float(rospy.get_param('/task2a_pointcloud_obstacles/self_filter_radius', .33)) > .23:
        parser.error('Mapper must use _self_filter_radius:=0.23')
    os.makedirs(args.diagnostics_dir, exist_ok=True)
    policy, update_coarse, pose, resources = create_session(args, SurfaceDQN)
    latest, contact, active = [None], [False], [False]
    contact_messages = []
    score = dict(initial=None, latest=None)
    def cloud(msg):
        latest[0] = msg
    def on_contact(msg):
        contact[0] = contact[0] or bool(msg.data)
    def on_contact_message(msg):
        contact_messages.append(msg.data)
    def on_status(msg):
        active[0] = any(s.status in (0, 1, 6, 7) for s in msg.status_list)
    def on_score(msg):
        if score['initial'] is None:
            score['initial'] = float(msg.data)
        score['latest'] = float(msg.data)
    subs = [rospy.Subscriber('/task2a_cloud/obstacles', PointCloud2, cloud,
                            queue_size=1, buff_size=16*1024*1024),
            rospy.Subscriber('/undesired_contact_detector/detect', Bool, on_contact, queue_size=1),
            rospy.Subscriber('/undesired_contact_detector/detect_msg', String, on_contact_message, queue_size=10),
            rospy.Subscriber('/move_base/status', GoalStatusArray, on_status, queue_size=1),
            rospy.Subscriber('/score', Float32, on_score, queue_size=1)]
    publisher = None
    result = dict(internal_goal_reached=False, official_success_verified=False,
                  mode='forward_first', phases=[], commands=[])
    last_stamp = [None]
    def refresh():
        if contact[0]:
            raise RuntimeError('Contact detected; stopped')
        if active[0]:
            raise RuntimeError('Another move_base goal is active')
        msg = latest[0]
        if msg is None:
            raise RuntimeError('Waiting for pointcloud')
        age = (rospy.Time.now()-msg.header.stamp).to_sec()
        if not -.5 <= age <= 3:
            raise RuntimeError('Pointcloud stale: %.3f s' % age)
        # Keep the servo loop light: rebuild geometry only for a new cloud.
        if last_stamp[0] != msg.header.stamp:
            update_coarse()
            policy.refresh(msg)
            last_stamp[0] = msg.header.stamp

    try:
        status = rospy.wait_for_message('/move_base/status', GoalStatusArray, timeout=5)
        on_status(status)
        deadline = time.monotonic()+15
        while True:
            try:
                refresh()
                position, yaw = pose()
                break
            except Exception:
                if contact[0] or active[0] or time.monotonic() >= deadline:
                    raise
                time.sleep(.1)
        if args.seed_run:
            seed = os.path.expanduser(args.seed_run)
            with open(os.path.join(seed, 'summary.json')) as f:
                previous = json.load(f)['pose_diagnostic']
            now = policy.pose_diagnostic
            angle = math.radians(now['yaw_map_odom_deg']-previous['yaw_map_odom_deg'])
            if (np.linalg.norm(position-np.asarray(previous['xy_map'])) > .10 or
                    np.linalg.norm(np.asarray(now['map_odom_xy'])-previous['map_odom_xy']) > .02 or
                    abs(math.atan2(math.sin(angle), math.cos(angle))) > math.radians(1) or
                    rospy.Time.now().to_sec() < previous['tf_stamp']):
                raise RuntimeError('Saved scene no longer matches this stopped pose/TF; seed refused')
            saved = np.load(os.path.join(seed, 'stopped_surfaces.npz'))['surfaces']
            t = policy.tf_buffer.lookup_transform('odom', 'map', rospy.Time(0), rospy.Duration(.2)).transform
            q, tr = t.rotation, t.translation
            r = rotation_matrix([q.x, q.y, q.z, q.w])
            points = saved @ r[:2, :2].T+[tr.x, tr.y]
            policy.memory.update(map(tuple, np.rint(points/.002).astype(int)))
            policy.refresh(latest[0])
        goal = np.array([args.goal_x, args.goal_y])
        # TASK2A demo change: use the actual heading, not map +Y, for forward.
        direction = np.array([math.cos(yaw), math.sin(yaw)])
        waypoints = forward_waypoints(position, yaw, goal)
        previous = position.copy()
        reports = []
        for i, target in enumerate(waypoints):
            # A preflight pass did not prevent the previous physical contact.
            # Do not relax the surface margin, including for the initial move.
            reports.append(segment_report(policy, previous, target))
            previous = target
        result.update(initial_position=position.tolist(), initial_yaw_deg=math.degrees(yaw),
                      waypoints=[p.tolist() for p in waypoints],
                      preflight=reports, preview=not args.execute)
        rospy.loginfo('FIXED ROUTE PREFLIGHT: %s', json.dumps(reports))
        if any(report['rejected_by'] for report in reports):
            raise RuntimeError('Fixed route intersects measured geometry; no motion sent')
        if not args.execute:
            return 0
        publisher = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
        deadline = time.monotonic()+3
        while publisher.get_num_connections() == 0:
            if time.monotonic() >= deadline or rospy.is_shutdown():
                raise RuntimeError('No command_velocity subscriber')
            time.sleep(.05)
        initial_tf = dict(policy.pose_diagnostic)
        reference_yaw = math.radians(initial_tf['yaw_odom_deg'])
        for i, target in enumerate(waypoints):
            refresh()
            start, _ = pose()
            report = segment_report(policy, start, target)
            if report['rejected_by']:
                raise RuntimeError('Updated geometry blocks phase %d' % i)
            phase = dict(index=i, target=target.tolist(), start=start.tolist())
            result['phases'].append(phase)
            rospy.loginfo('FIXED PHASE %d: target=%s', i, target.tolist())
            forward = i == 0
            axis = 0
            tolerance = .008
            deadline = time.monotonic()+90
            while True:
                if rospy.is_shutdown() or time.monotonic() >= deadline:
                    raise RuntimeError('Fixed phase timed out or ROS shut down')
                refresh()
                current, map_yaw = pose()
                diag = policy.pose_diagnostic
                shift = np.linalg.norm(np.asarray(diag['map_odom_xy'])-initial_tf['map_odom_xy'])
                turn = math.radians(diag['yaw_map_odom_deg']-initial_tf['yaw_map_odom_deg'])
                if shift > .02 or abs(math.atan2(math.sin(turn), math.cos(turn))) > math.radians(1):
                    raise RuntimeError('Localization changed during fixed route')
                if np.linalg.norm(current-start) > np.linalg.norm(target-start)+.10:
                    raise RuntimeError('Left bounded fixed phase')
                if not policy.known_clear(current, target):
                    raise RuntimeError('Updated pointcloud blocks fixed segment')
                error = target-current
                along_remaining = float(np.dot(error, direction))
                cross_error = abs(float(np.cross(direction, current-position)))
                if forward and cross_error > .01:
                    raise RuntimeError('Forward route drift exceeded 1 cm; no sideways correction sent')
                if (forward and along_remaining <= tolerance) or (not forward and np.linalg.norm(error) <= tolerance):
                    publisher.publish(Twist())
                    phase['final_position'] = current.tolist()
                    break
                heading = math.radians(diag['yaw_odom_deg'])
                heading_error = math.atan2(math.sin(reference_yaw-heading), math.cos(reference_yaw-heading))
                if abs(heading_error) > math.radians(3 if forward else 8):
                    raise RuntimeError('Heading drift exceeded fixed phase limit')
                velocity = np.clip(error*.8, -.01, .01)
                maximum = .012 if i == 0 else .02
                along = min(maximum, max(.004, abs(error[axis])*.6))
                velocity[axis] = math.copysign(along, error[axis]) if abs(error[axis]) > tolerance/2 else 0.
                c, s = math.cos(map_yaw), math.sin(map_yaw)
                command = Twist()
                command.linear.x = float(c*velocity[0]+s*velocity[1])
                command.linear.y = float(-s*velocity[0]+c*velocity[1])
                if forward:
                    # TASK2A demo change: forward phase never strafes or turns.
                    command.linear.x = min(.015, max(.004, along_remaining*.6))
                    command.linear.y = 0.
                elif abs(heading_error) > math.radians(2):
                    command.angular.z = float(np.clip(heading_error*.6, -.04, .04))
                result['commands'].append(dict(position=current.tolist(), phase=i,
                    linear_x=command.linear.x, linear_y=command.linear.y, angular_z=command.angular.z))
                publisher.publish(command)
                rospy.loginfo_throttle(2., 'FIXED phase=%d position=%s error=%.3f clearance=%.3f',
                    i, current.tolist(), np.linalg.norm(error), policy.tree.query(current)[0])
                time.sleep(.05)
            for _ in range(6):
                publisher.publish(Twist())
                time.sleep(.05)
        final, _ = pose()
        result.update(final_position=final.tolist(), remaining=float(np.linalg.norm(final-goal)))
        result['internal_goal_reached'] = result['remaining'] <= .12
        rospy.loginfo('FIXED ROUTE FINISHED: %s', json.dumps(result))
        return 0 if result['internal_goal_reached'] else 1
    except Exception as error:
        result['error'] = str(error)
        rospy.logerr('FIXED ROUTE STOPPED: %s', error)
        return 1
    finally:
        if publisher is not None:
            for _ in range(6):
                publisher.publish(Twist())
                time.sleep(.05)
            publisher.unregister()
        for sub in subs+list(resources[1:]):
            sub.unregister()
        result['contact_detected'] = contact[0]
        result['contact_messages'] = contact_messages
        result['pose_diagnostic'] = dict(policy.pose_diagnostic)
        if policy.tree is not None:
            np.savez_compressed(os.path.join(args.diagnostics_dir, 'stopped_surfaces.npz'),
                                surfaces=policy.tree.data)
        result['score'] = score
        with open(os.path.join(args.diagnostics_dir, 'summary.json'), 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    raise SystemExit(main())
