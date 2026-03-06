#!/bin/bash
# Run PPO for all non-100% tasks in parallel, 4 at a time

LOGDIR="/234/outputs/ppo_logs"
mkdir -p $LOGDIR

echo "Starting batch 1: tasks 0, 1, 5, 9"
python3 /234/scripts/residual_ppo_single.py --task 0 > $LOGDIR/task0.log 2>&1 &
python3 /234/scripts/residual_ppo_single.py --task 1 > $LOGDIR/task1.log 2>&1 &
python3 /234/scripts/residual_ppo_single.py --task 5 > $LOGDIR/task5.log 2>&1 &
python3 /234/scripts/residual_ppo_single.py --task 9 > $LOGDIR/task9.log 2>&1 &

wait
echo "Batch 1 done. Starting batch 2: tasks 3, 6, 7, 8"

python3 /234/scripts/residual_ppo_single.py --task 3 > $LOGDIR/task3.log 2>&1 &
python3 /234/scripts/residual_ppo_single.py --task 6 > $LOGDIR/task6.log 2>&1 &
python3 /234/scripts/residual_ppo_single.py --task 7 > $LOGDIR/task7.log 2>&1 &
python3 /234/scripts/residual_ppo_single.py --task 8 > $LOGDIR/task8.log 2>&1 &

wait
echo "All done!"
