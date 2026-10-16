#!/usr/bin/env python3
"""Prepare a freshly reset Gazebo robot at standby_2a; default is preview."""
import argparse
import json
import math
import os
import time


def wait_action(client, seconds):
    import rospy
    from actionlib_msgs.msg import GoalStatus
    deadline = time.monotonic()+seconds
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            return
        if state in (GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED,
                     GoalStatus.RECALLED, GoalStatus.LOST):
            raise RuntimeError(client.get_goal_status_text())
        time.sleep(.05)
    client.cancel_goal()
    raise RuntimeError('Preparation action timed out or ROS shut down')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    import rospy
    import rospkg
    import actionlib
    import tf2_ros
    from actionlib_msgs.msg import GoalStatusArray
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectoryPoint
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
    rospy.init_node('task2a_prepare')
    root = rospkg.RosPack().get_path('wrs_algorithm')
    with open(os.path.join(root, 'config', 'poses.json')) as f:
        joints = json.load(f)['look_at_near_floor']
    with open(os.path.join(root, 'config', 'coordinates.json')) as f:
        target = json.load(f)['positions']['standby_2a']
    rospy.loginfo('Prepare floor-view posture and standby_2a=%s execute=%s', target, args.execute)
    if not args.execute:
        return 0
    if not rospy.get_param('/use_sim_time', False):
        raise RuntimeError('This preparation command is for Gazebo')
    status = rospy.wait_for_message('/move_base/status', GoalStatusArray, timeout=5)
    if any(item.status in (0, 1, 6, 7) for item in status.status_list):
        raise RuntimeError('Another move_base goal is active')
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
    transform = buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(5))
    if transform.transform.translation.y > 2.1:
        raise RuntimeError('Robot is already in the obstacle area: reset the simulator before preparation')
    clients = []
    try:
        for group, names in [('arm', ['arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint',
                                     'wrist_flex_joint', 'wrist_roll_joint']),
                             ('head', ['head_pan_joint', 'head_tilt_joint'])]:
            client = actionlib.SimpleActionClient('/hsrb/%s_trajectory_controller/follow_joint_trajectory' % group,
                                                  FollowJointTrajectoryAction)
            if not client.wait_for_server(rospy.Duration(5)):
                raise RuntimeError('Missing '+group+' trajectory controller')
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = names
            goal.trajectory.points = [JointTrajectoryPoint(
                positions=[joints[n] for n in names], time_from_start=rospy.Duration(3))]
            client.send_goal(goal)
            clients.append(client)
            wait_action(client, 20)
        client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        if not client.wait_for_server(rospy.Duration(5)):
            raise RuntimeError('move_base unavailable')
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x, goal.target_pose.pose.position.y = target[:2]
        goal.target_pose.pose.orientation.z = math.sin(math.radians(target[2])/2)
        goal.target_pose.pose.orientation.w = math.cos(math.radians(target[2])/2)
        client.send_goal(goal)
        clients.append(client)
        wait_action(client, 120)
        rospy.loginfo('STANDBY READY. Start the point-cloud mapper, then preview the precise runner.')
        return 0
    finally:
        for client in clients:
            if client.get_state() in (0, 1, 6, 7):
                client.cancel_goal()


if __name__ == '__main__':
    raise SystemExit(main())
