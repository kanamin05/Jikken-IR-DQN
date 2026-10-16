#!/bin/bash
cd /home/developer/workspace/task2a_dqn_starter || exit 1

.venv/bin/python -u train_dqn.py --obstacles 2 --steps 10000
training_exit_code=$?

echo "TRAIN_EXIT_CODE=$training_exit_code"
exit "$training_exit_code"
