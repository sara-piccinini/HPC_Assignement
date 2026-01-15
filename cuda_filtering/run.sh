#!/usr/bin/env bash
set -euo pipefail

# run.sh (no CMake, no OpenCV C++ required)
# Uses python+cv2 for resizing, then runs ./filter (CUDA) on sized PNGs.
#
# Usage:
#   ./run.sh <image_or_dir> [reps]
#
# Examples:
#   ./run.sh images 10
#   ./run.sh pexels-christian-heitz.jpg 20

TARGET="${1:-}"
REPS="${2:-10}"

if [[ -z "${TARGET}" ]]; then
  echo "Usage: $0 <image_or_dir> [reps]"
  exit 1
fi

OUTDIR="out"
SIZEDIR="sized"
mkdir -p "${OUTDIR}" "${SIZEDIR}"

FILTER_EXE="./filter"

# block sweep (keep BX*BY <= 1024)
BLOCKS=(
  "8x8" "16x8" "16x16"
  "32x8" "32x16" "32x32"
  "8x16" "8x32" "16x32"
)

# exact ES2 sizes
declare -A SIZES
SIZES["4k"]="3840x2160"
SIZES["8k"]="7680x4320"
SIZES["16k"]="15360x8640"

echo "[info] TARGET=${TARGET}"
echo "[info] REPS=${REPS}"
echo "[info] OUTDIR=${OUTDIR}"
echo "[info] SIZEDIR=${SIZEDIR}"
echo "[info] FILTER_EXE=${FILTER_EXE}"

# ---- Ensure filter exists ----
if [[ ! -x "${FILTER_EXE}" ]]; then
  echo "[error] ${FILTER_EXE} not found or not executable."
  echo "        Build it first (job.sbatch does this), or run: nvcc -O3 -std=c++17 filter.cu -o filter"
  exit 2
fi

# ---- Require python3 + cv2 ----
if ! python3 -c "import cv2" >/dev/null 2>&1; then
  echo "[error] python3 + cv2 not available (needed for resizing)."
  exit 3
fi

# ---- Collect input images ----
IMAGES=()
if [[ -d "${TARGET}" ]]; then
  while IFS= read -r -d '' f; do IMAGES+=("$f"); done < <(
    find "${TARGET}" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | sort -z
  )
else
  IMAGES+=("${TARGET}")
fi

if [[ "${#IMAGES[@]}" -eq 0 ]]; then
  echo "[error] no images found."
  exit 4
fi

# ---- Resize helper (exact WxH) ----
resize_exact () {
  local inpath="$1"
  local outpath="$2"
  local w="$3"
  local h="$4"

  python3 - <<'PY' "$inpath" "$outpath" "$w" "$h"
import sys, cv2
inp, outp, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
img = cv2.imread(inp, cv2.IMREAD_COLOR)
if img is None:
    raise SystemExit(f"Cannot read {inp}")
res = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
ok = cv2.imwrite(outp, res)
if not ok:
    raise SystemExit(f"Cannot write {outp}")
PY
}

# ---- GPU sampler (nvidia-smi) ----
start_gpu_sampler () {
  local outfile="$1"
  echo "timestamp,util_gpu_pct,util_mem_pct,mem_used_MiB,mem_total_MiB,power_W" > "$outfile"
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits 2>/dev/null || true
      sleep 1
    done
  ) >> "$outfile" &
  echo $!
}

stop_gpu_sampler () {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

# ---- Results CSV ----
CSV="${OUTDIR}/results.csv"
echo "input_base,variant,width,height,blockX,blockY,reps,h2d_ms,kernel_ms_avg,d2h_ms,total_ms,output" > "${CSV}"

# ---- Main loop ----
for img in "${IMAGES[@]}"; do
  echo "[info] base input: ${img}"
  base="$(basename "$img")"
  stem="${base%.*}"

  # Create exact sized images
  for tag in 4k 8k 16k; do
    dims="${SIZES[$tag]}"
    w="${dims%x*}"
    h="${dims#*x}"
    out_s="${SIZEDIR}/${stem}_${tag}.png"
    echo "[info] resizing -> ${out_s} (${w}x${h})"
    resize_exact "${img}" "${out_s}" "${w}" "${h}"
  done

  # Run per size with resource logging
  for tag in 4k 8k 16k; do
    dims="${SIZES[$tag]}"
    w="${dims%x*}"
    h="${dims#*x}"
    in_s="${SIZEDIR}/${stem}_${tag}.png"

    gpu_log="${OUTDIR}/gpu_usage_${tag}.csv"
    echo "[info] starting GPU sampler -> ${gpu_log}"
    sampler_pid="$(start_gpu_sampler "${gpu_log}")"
    trap 'stop_gpu_sampler "${sampler_pid:-}"; exit 99' INT TERM

    for b in "${BLOCKS[@]}"; do
      bx="${b%x*}"
      by="${b#*x}"
      if (( bx * by > 1024 )); then
        echo "[warn] skipping invalid block ${bx}x${by} (>1024 threads)"
        continue
      fi

      out_img="${OUTDIR}/${stem}_${tag}_b${bx}x${by}.png"
      echo "[run] size=${tag} (${w}x${h}) block=${bx}x${by} reps=${REPS}"

      line="$(${FILTER_EXE} "${in_s}" "${out_img}" "${bx}" "${by}" "${REPS}")"
      # RESULT,input,output,width,height,blockX,blockY,reps,h2d,kernel_avg,d2h,total
      IFS=',' read -r _ inpath outpath rw rh blockX blockY reps h2d kernel d2h total <<< "${line}"

      echo "${img},${tag},${rw},${rh},${blockX},${blockY},${reps},${h2d},${kernel},${d2h},${total},${outpath}" >> "${CSV}"
    done

    echo "[info] stopping GPU sampler (${sampler_pid})"
    stop_gpu_sampler "${sampler_pid}"
    unset sampler_pid
    trap - INT TERM
  done
done

echo "[done] results -> ${CSV}"
echo "[done] gpu logs -> ${OUTDIR}/gpu_usage_4k.csv, ${OUTDIR}/gpu_usage_8k.csv, ${OUTDIR}/gpu_usage_16k.csv"


