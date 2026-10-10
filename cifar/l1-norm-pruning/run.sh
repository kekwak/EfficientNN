#!/usr/bin/env bash
# Полный цикл воспроизведения (rethinking-network-pruning / l1-norm-pruning)
# Запуск: bash run_all.sh 2>&1 | tee run_all.log
set -uo pipefail

DATASET=${DATASET:-cifar10}
SEEDS=(1275 2024 3407)                          # 3 фиксированных сида
PY=${PY:-python}
RUN_ROOT=${RUN_ROOT:-runs_$(date +%m%d)}

# arch : depth : prune-скрипт : доп. флаги (resnet110 не берём)
CONFIGS=(
  "vgg:16:vggprune.py:"
  "resnet:56:res56prune.py:-v A"
)

# какой чекпоинт оставил baseline
pick_ckpt() {
  for f in checkpoint.pth.tar model_best.pth.tar; do
    [ -f "$1/$f" ] && { echo "$1/$f"; return 0; }
  done
  ls -t "$1"/*.pth.tar 2>/dev/null | head -1
}

# дать --seed только если скрипт его понимает
seed_flag() {
  $PY "$1" --help 2>&1 | grep -q -- '--seed' && echo "--seed $2" || echo ""
}

# шаг: если рядом лежит DONE — пропускаем (можно перезапускать скрипт)
run_step() {
  local dir=$1 log=$2; shift 2
  if [ -f "$dir/DONE" ]; then echo "-- пропуск (готово): $log"; return 0; fi
  echo ">>> $log"
  if "$@" 2>&1 | tee "$log"; then
    touch "$dir/DONE"
  else
    echo "!!! FAILED: $log" | tee -a "$RUN_ROOT/errors.txt"
    return 1
  fi
}

for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r arch depth pscript pextra <<< "$cfg"
  for seed in "${SEEDS[@]}"; do
    D="$RUN_ROOT/${arch}${depth}/seed${seed}"
    mkdir -p "$D"/{baseline,prune,finetune,scratchE,scratchB}
    echo "===== ${arch}${depth} | seed=${seed} ====="

    run_step "$D/baseline" "$D/baseline.log" \
      $PY main.py --dataset $DATASET --arch $arch --depth $depth --seed $seed --save "$D/baseline" || continue

    BASE=$(pick_ckpt "$D/baseline")
    run_step "$D/prune" "$D/prune.log" \
      $PY $pscript --dataset $DATASET $pextra --model "$BASE" --save "$D/prune" || continue

    PRUNED="$D/prune/pruned.pth.tar"
    run_step "$D/finetune" "$D/finetune.log" \
      $PY main_finetune.py --refine "$PRUNED" --dataset $DATASET --arch $arch --depth $depth \
      $(seed_flag main_finetune.py $seed) --save "$D/finetune"

    run_step "$D/scratchE" "$D/scratchE.log" \
      $PY main_E.py --scratch "$PRUNED" --dataset $DATASET --arch $arch --depth $depth \
      $(seed_flag main_E.py $seed) --save "$D/scratchE"

    run_step "$D/scratchB" "$D/scratchB.log" \
      $PY main_B.py --scratch "$PRUNED" --dataset $DATASET --arch $arch --depth $depth \
      $(seed_flag main_B.py $seed) --save "$D/scratchB"
  done
done

echo "=== Итоги (best accuracy) ===" | tee "$RUN_ROOT/summary.txt"
grep -rih "best" "$RUN_ROOT" --include="*.log" | tee -a "$RUN_ROOT/summary.txt"
