#!/usr/bin/env python3
"""Kanami: ONE DQN action through move_base in Gazebo, then exit.

Uses existing exported weights and live point-cloud grid; no detector labels.
Run with --execute to send one goal, otherwise prints the candidate only.
"""
import os
import math
import time
import argparse
import numpy as np
from task2a_cloud_infer import CloudDQN


def validate_map(policy, msg, check_static=False):
    q = msg.info.origin.orientation
    if (msg.header.frame_id.lstrip('/') != 'map' or
        (msg.info.height, msg.info.width) != policy.shape or
        abs(msg.info.resolution-policy.resolution) > 1e-6 or
        not np.allclose([msg.info.origin.position.x, msg.info.origin.position.y],
                        policy.origin, atol=1e-6) or
        abs(q.x)+abs(q.y)+abs(q.z) > 1e-6 or abs(abs(q.w)-1) > 1e-6):
        raise ValueError('Map metadata differs from training')
    raw = np.asarray(msg.data).reshape(policy.shape)
    if check_static and not np.array_equal((raw >= 0) & (raw <= 19), policy.data['free']):
        raise ValueError('Live static map differs from training')
    return raw


def check_path(policy, start, target, path, max_length):
    """Check every segment, including the actual start and final target."""
    previous = np.asarray(start)
    total = 0.
    for point in list(path)+[target]:
        point = np.asarray(point)
        if not np.isfinite(point).all():
            raise ValueError('Invalid move_base plan coordinate')
        total += np.linalg.norm(point-previous)
        if not policy.known_clear(previous, point):
            raise ValueError('move_base plan enters known blocked cells or leaves ROI')
        previous = point
    if total > max_length:
        raise ValueError('Plan detours too far for a one-step check: %.3f m' % total)


def run_one_step(args):
    import rospy
    import tf2_ros
    import actionlib
    from nav_msgs.msg import OccupancyGrid
    from nav_msgs.srv import GetPlan, GetPlanRequest
    from std_msgs.msg import Bool
    from geometry_msgs.msg import PoseStamped
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
    from actionlib_msgs.msg import GoalStatus

    policy = CloudDQN(os.path.expanduser(args.weights))
    validate_map(policy, rospy.wait_for_message('/map', OccupancyGrid, timeout=15), True)
    for key in ('robot_radius', 'margin'):
        value = float(rospy.get_param('/task2a_pointcloud_obstacles/'+key, float(policy.data[key])))
        if abs(value-float(policy.data[key])) > 1e-6:
            raise ValueError('Mapper and model disagree on '+key)
    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30))
    listener = tf2_ros.TransformListener(buffer)
    latest, validity = [None, 0.], [False, 0.]

    def receive(msg):
        latest[:] = [msg, time.monotonic()]

    def receive_valid(msg):
        validity[:] = [msg.data, time.monotonic()]

    rospy.Subscriber('/task2a_cloud/occupied', OccupancyGrid, receive, queue_size=1)
    rospy.Subscriber('/task2a_cloud/valid', Bool, receive_valid, queue_size=1)

    def update_cloud():
        msg, received = latest[:]
        valid, valid_received = validity[:]
        now = time.monotonic()
        if (msg is None or not valid or now-received > 3 or now-valid_received > 3):
            raise ValueError('Point-cloud map unavailable/invalid or updates stopped')
        age = (rospy.Time.now()-msg.header.stamp).to_sec()
        if msg.header.stamp == rospy.Time(0) or age < -.5 or age > 3:
            raise ValueError('Point-cloud map stale: age=%.2f s' % age)
        policy.update(validate_map(policy, msg))

    def pose():
        stamped = buffer.lookup_transform('map', args.base_frame,
                                          rospy.Time(0), rospy.Duration(.5))
        age = (rospy.Time.now()-stamped.header.stamp).to_sec()
        if age < -.5 or age > 3:
            raise ValueError('Robot TF stale: age=%.2f s' % age)
        p, q = stamped.transform.translation, stamped.transform.rotation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        if not np.isfinite([p.x, p.y, yaw]).all():
            raise ValueError('Invalid robot TF')
        return np.array([p.x, p.y]), yaw

    ready_until = time.monotonic()+15
    while True:
        if rospy.is_shutdown():
            return False
        try:
            update_cloud()
            start, yaw = pose()
            break
        except Exception as error:
            if time.monotonic() >= ready_until:
                raise RuntimeError('Initialization timed out: '+str(error))
            rospy.logwarn_throttle(3, 'Waiting: %s' % error)
            time.sleep(.1)
    yaw_error = math.atan2(math.sin(yaw-math.pi/2), math.cos(yaw-math.pi/2))
    if abs(yaw_error) > math.radians(15):
        raise ValueError('Expected yaw near +90 degrees; got %.1f' % math.degrees(yaw))
    if np.linalg.norm(start-policy.data['goal']) <= .12:
        rospy.loginfo('Already near Task2a goal; no motion sent.')
        return True
    observation, _ = policy.observation(start)
    values = policy.qvalues(observation)
    raw = int(np.argmax(values))
    chosen = None
    for candidate in np.argsort(-values):
        target = start+policy.data['actions'][candidate]*float(policy.data['step_size'])
        if policy.known_clear(start, target):
            chosen = int(candidate)
            break
    if chosen is None:
        raise ValueError('All eight actions blocked by known geometry')
    target = start+policy.data['actions'][chosen]*float(policy.data['step_size'])
    step_size = float(policy.data['step_size'])
    if not (0 < step_size <= .15+1e-6):
        raise ValueError('This script permits step_size at most 0.15 m')
    rospy.loginfo('ONE STEP: start=%s yaw=%.1f deg raw=%d selected=%d target=%s Q=%s',
                  start, math.degrees(yaw), raw, chosen, target, np.array2string(values, precision=2))
    if not args.execute:
        rospy.loginfo('Preview complete; no goal sent. --execute enables ONE goal.')
        return True

    client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    if not client.wait_for_server(rospy.Duration(5)):
        raise RuntimeError('move_base unavailable')

    def pose_message(xy):
        msg = PoseStamped()
        msg.header.frame_id, msg.header.stamp = 'map', rospy.Time.now()
        msg.pose.position.x, msg.pose.position.y = float(xy[0]), float(xy[1])
        msg.pose.orientation.z = msg.pose.orientation.w = math.sqrt(.5)
        return msg

    rospy.wait_for_service('/move_base/make_plan', timeout=5)
    response = rospy.ServiceProxy('/move_base/make_plan', GetPlan)(
        GetPlanRequest(start=pose_message(start), goal=pose_message(target), tolerance=0.))
    if not response.plan.poses:
        raise ValueError('move_base returned an empty plan')
    plan_frame = response.plan.header.frame_id.lstrip('/')
    if plan_frame and plan_frame != 'map':
        raise ValueError('Unexpected plan frame: '+plan_frame)
    # Some ROS1 planners fill PoseStamped headers but leave Path.header empty.
    # Accept that case only when each pose resolves explicitly to map.
    for waypoint in response.plan.poses:
        frame = waypoint.header.frame_id.lstrip('/') or plan_frame
        if frame != 'map':
            raise ValueError('Plan waypoint frame is not confirmed as map: %r' % frame)
    path = [np.array([p.pose.position.x, p.pose.position.y]) for p in response.plan.poses]
    update_cloud()
    current, _ = pose()
    if np.linalg.norm(current-start) > .03:
        raise ValueError('Robot moved while preparing the goal; run again')
    check_path(policy, current, target, path, max_length=2*step_size+.1)
    goal = MoveBaseGoal(target_pose=pose_message(target))
    sent = False
    try:
        rospy.loginfo('Sending ONE move_base goal, maximum wait %.1f wall seconds.', args.timeout)
        client.send_goal(goal)
        sent = True
        deadline = time.monotonic()+args.timeout
        while not rospy.is_shutdown():
            update_cloud()
            current, current_yaw = pose()
            if np.linalg.norm(current-start) > step_size+.10:
                raise RuntimeError('Robot traveled beyond the one-step area')
            yaw_error = math.atan2(math.sin(current_yaw-math.pi/2), math.cos(current_yaw-math.pi/2))
            if abs(yaw_error) > math.radians(20):
                raise RuntimeError('Robot yaw deviated: start=%.1f current=%.1f deg; xy=%s' %
                                   (math.degrees(yaw), math.degrees(current_yaw), current))
            if not policy.known_clear(current, target):
                raise RuntimeError('Updated cloud blocks current position/remaining segment')
            check_path(policy, start, target, path, max_length=2*step_size+.1)
            state = client.get_state()
            if state == GoalStatus.SUCCEEDED:
                if np.linalg.norm(current-target) > .10:
                    raise RuntimeError('move_base succeeded but target error exceeds 0.10 m')
                rospy.loginfo('ONE STEP COMPLETE: final=%s displacement=%.3f m target_error=%.3f m. Exiting.',
                              current, np.linalg.norm(current-start), np.linalg.norm(current-target))
                return True
            if state in (GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED,
                         GoalStatus.RECALLED, GoalStatus.LOST):
                raise RuntimeError('move_base ended with state=%d: %s' % (state, client.get_goal_status_text()))
            if time.monotonic() >= deadline:
                raise RuntimeError('One-step motion timed out')
            time.sleep(.05)
        return False
    finally:
        if sent and client.get_state() not in (GoalStatus.SUCCEEDED, GoalStatus.ABORTED,
                GoalStatus.REJECTED, GoalStatus.PREEMPTED, GoalStatus.RECALLED):
            client.cancel_goal()
            # Wait briefly for acknowledgement; never cancel another node's goals.
            until = time.monotonic()+2
            while time.monotonic() < until and client.get_state() in (
                    GoalStatus.PENDING, GoalStatus.ACTIVE, GoalStatus.PREEMPTING, GoalStatus.RECALLING):
                time.sleep(.05)
            rospy.logwarn('One-step goal canceled; move_base state=%d', client.get_state())


def main():
    import rospy
    rospy.init_node('task2a_cloud_step')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--tune-dwa', action='store_true',
                        help='Temporarily permit backward/sideways motion at <=0.10 m/s; restore on exit')
    parser.add_argument('--weights', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        'task2a_policy_numpy.npz'))
    parser.add_argument('--base-frame', default='base_footprint')
    parser.add_argument('--timeout', type=float, default=20.)
    args = parser.parse_args(rospy.myargv()[1:])
    if not (0 < args.timeout <= 30):
        parser.error('--timeout must be >0 and <=30 seconds')
    tuning = None
    original = None
    try:
        if args.execute and args.tune_dwa:
            from dynamic_reconfigure.client import Client
            planner = rospy.get_param('/move_base/base_local_planner')
            if planner != 'dwa_local_planner/DWAPlannerROS':
                raise ValueError('--tune-dwa expects DWAPlannerROS, got '+planner)
            tuning = Client('/move_base/DWAPlannerROS', timeout=5)
            settings = dict(min_vel_x=-.10, max_vel_x=.10,
                            min_vel_y=-.10, max_vel_y=.10,
                            min_vel_trans=.02, max_vel_trans=.10,
                            min_vel_theta=.05, max_vel_theta=.30,
                            vx_samples=11, vy_samples=11, vth_samples=11,
                            twirling_scale=1., forward_point_distance=0.)
            before = tuning.get_configuration(timeout=5)
            if not all(key in before for key in settings):
                raise ValueError('DWA configuration missing required settings')
            original = {key: before[key] for key in settings}
            actual = tuning.update_configuration(settings)
            if any(abs(float(actual[key])-value) > 1e-6 for key, value in settings.items()):
                raise ValueError('DWA did not accept requested settings')
            rospy.loginfo('Temporary DWA settings: backwards/lateral enabled, speed <=0.10 m/s, rotation <=0.30 rad/s.')
        return 0 if run_one_step(args) else 1
    except Exception as error:
        rospy.logerr('ONE STEP STOPPED: %s', error)
        return 1
    finally:
        if tuning is not None and original is not None:
            try:
                tuning.update_configuration(original)
                rospy.loginfo('Original DWA settings restored.')
            except Exception as error:
                rospy.logerr('Could not restore DWA settings: %s; original=%s', error, original)


if __name__ == '__main__':
    raise SystemExit(main())
