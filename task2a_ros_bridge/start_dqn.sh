#!/bin/bash
set -eu

# Same ROS setup as the existing start_all.sh. Do not activate the learning venv.
unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash

pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        kill -INT "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

roslaunch hsr_perception pcd_to_tf.launch &
pids+=("$!")
roslaunch wrs_detector frcnn_finetuned.launch &
pids+=("$!")
roslaunch wrs_algorithm task2a_dqn.launch "$@"
