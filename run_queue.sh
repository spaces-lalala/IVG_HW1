#!/bin/bash
cd /workspace/mamba_logo/generate/IVG_HW1/image_diffusion_todo
LOG=/workspace/mamba_logo/generate/IVG_HW1/queue.log

echo "$(date '+%F %T') waiting for running train.py to finish" >> $LOG
while pgrep -f "train.py --mode" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') GPU free, starting queue" >> $LOG

for cfg in "cosine noise" "linear x0" "linear mean"; do
    set -- $cfg
    echo "$(date '+%F %T') START  mode=$1 predictor=$2" >> $LOG
    uv run python train.py --mode "$1" --predictor "$2" --log_interval 2000
    echo "$(date '+%F %T') FINISH mode=$1 predictor=$2 exit=$?" >> $LOG
done
echo "$(date '+%F %T') ALL DONE" >> $LOG
