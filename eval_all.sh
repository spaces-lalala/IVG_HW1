#!/bin/bash
set -u
ROOT=/workspace/mamba_logo/generate/IVG_HW1
CDIR=$ROOT/image_diffusion_todo
LOG=$ROOT/eval.log
OUT=$ROOT/report_assets
RES=$ROOT/fid_results.md
cd "$CDIR" || exit 1

log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

log "=== eval_all start ==="
log "waiting for any running train.py"
while pgrep -f "train.py --mode" > /dev/null; do sleep 60; done
log "no training running; proceeding"

mkdir -p "$OUT"
: > "$RES"
{
  echo "# FID results"
  echo
  echo "| run | mode | predictor | ckpt_dir | last_log_step | traj_n | ckpt_bytes | FID |"
  echo "|---|---|---|---|---|---|---|---|"
} >> "$RES"

find_dir(){   # $1=mode $2=predictor -> dir that actually holds a ckpt
  for d in $(ls -d results/predictor_$2/beta_$1/*/ 2>/dev/null | sort); do
    [ -f "${d}last.ckpt" ] && echo "${d%/}"
  done | tail -1
}

eval_run(){
  local tag=$1 mode=$2 pred=$3
  local dir; dir=$(find_dir "$mode" "$pred")
  if [ -z "$dir" ]; then
    log "!! MISSING checkpoint for $tag ($mode/$pred)"
    echo "| $tag | $mode | $pred | (missing) | - | - | - | **NO CKPT** |" >> "$RES"
    return
  fi
  local ck="$dir/last.ckpt"
  local step traj bytes
  step=$(ls "$dir" | grep -oP 'step=\K[0-9]+' | sort -n | tail -1)
  traj=$(ls "$dir" | grep -c 'traj.png')
  bytes=$(stat -c%s "$ck")
  log "[$tag] dir=$dir step=$step traj=$traj bytes=$bytes"

  # ---- 500 samples for FID (skip if already complete)
  local sd="samples/$tag"
  local have=0; [ -d "$sd" ] && have=$(ls "$sd" 2>/dev/null | wc -l)
  if [ "$have" -lt 500 ]; then
    log "[$tag] sampling 500 ..."
    uv run python sampling.py --ckpt_path "$ck" --save_dir "$sd" >/dev/null 2>>"$LOG"; rc=$?
    log "[$tag] sampling rc=$rc"
  else
    log "[$tag] sampling skipped ($have images already)"
  fi

  # ---- FID
  local fid
  fid=$(uv run python fid/measure_fid.py data/afhq/eval "$sd" 2>>"$LOG" | grep -oP 'FID:\s*\K[0-9.]+')
  log "[$tag] FID=${fid:-FAILED}"

  # ---- 8 images + trajectory for the report
  log "[$tag] sampling 8 + traj ..."
  uv run python sampling.py --ckpt_path "$ck" --save_dir "samples/vis_$tag" \
      --num_samples 8 --save_traj >/dev/null 2>>"$LOG"

  # ---- collect small figures
  mkdir -p "$OUT/${tag}_samples8"
  cp "$dir/step=98000-traj.png" "$OUT/${tag}_traj_train.png"  2>/dev/null
  cp "$dir/loss.png"            "$OUT/${tag}_loss.png"        2>/dev/null
  cp "$dir/config.json"         "$OUT/${tag}_config.json"     2>/dev/null
  cp "samples/vis_${tag}_traj.png" "$OUT/${tag}_traj_sample.png" 2>/dev/null
  cp samples/vis_$tag/*.png     "$OUT/${tag}_samples8/"        2>/dev/null

  echo "| $tag | $mode | $pred | $dir | $step | $traj | $bytes | **${fid:-FAILED}** |" >> "$RES"
}

eval_run linear_noise linear noise
eval_run quad_noise   quad   noise
eval_run cosine_noise cosine noise
eval_run linear_x0    linear x0
eval_run linear_mean  linear mean

{
  echo
  echo "註：每個 run 皆為 100,000 steps、batch 16、log_interval 2000。"
  echo "last_log_step 應為 98000、traj_n 應為 50、ckpt_bytes 應為 235428141，"
  echo "三者皆符合才代表該 run 完整跑完。"
  echo
  echo "linear_noise 另有一次獨立抽樣得 FID 15.706（同一 checkpoint），"
  echo "用以估計抽樣變異約 0.4。"
  echo
  echo "產生時間：$(date '+%F %T')"
} >> "$RES"

log "=== all evaluations done ==="
cat "$RES" | tee -a "$LOG"

cd "$ROOT"
git add report_assets fid_results.md eval.log queue.log eval_all.sh 2>>"$LOG"
git commit -m "eval: FID + report figures for all 5 runs" >>"$LOG" 2>&1; rc=$?
log "git commit rc=$rc"
git push >>"$LOG" 2>&1; rc=$?
log "git push rc=$rc"
[ "$rc" -ne 0 ] && log "!! PUSH FAILED - 結果都在容器裡，回來手動 git push"
log "=== eval_all finished ==="
