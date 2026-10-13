#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PATH="$PWD/.venv/bin:/usr/local/cuda-13.1/bin:$PATH"
export TRITON_CACHE_DIR="$PWD/.cache/triton" FLASHINFER_WORKSPACE_BASE="$PWD/.cache/flashinfer"
export CUDA_HOME=/usr/local/cuda-13.1 MAX_JOBS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p outputs/lq-study/source outputs/lq-study/profiles
cp benchmark.py profile_components.py lq_optimizations.py lq_fp8.py validate_lq.py run_lq_study.sh outputs/lq-study/source/
cp config/head-policy.json outputs/lq-study/source/
for mode in baseline bf16_layout fp8_conv2 combined; do
    if [[ -f "outputs/lq-study/$mode/results.json" ]] && .venv/bin/python -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(not(r["status"]=="completed" and len(r["cases"])==6))' "outputs/lq-study/$mode/results.json"; then
        echo "REUSED_COMPLETED $mode"
        continue
    fi
    .venv/bin/python -u benchmark.py --attention headwise_bf16 --ffn nvfp4 --decoder fp8 --lq "$mode" --warmup \
        --output-dir "outputs/lq-study/$mode" > "logs/lq-$mode-generation.log" 2>&1
    echo "GENERATED $mode"
done
for repeat in 1 2 3; do
    case "$repeat" in
        1) modes='baseline bf16_layout fp8_conv2 combined' ;;
        2) modes='combined fp8_conv2 bf16_layout baseline' ;;
        3) modes='fp8_conv2 baseline combined bf16_layout' ;;
    esac
    for mode in $modes; do
        .venv/bin/python -u profile_components.py --attention headwise_bf16 --ffn nvfp4 --decoder fp8 --lq "$mode" \
            --warmup --detail --output "outputs/lq-study/profiles/$mode-$repeat.json" > "logs/lq-$mode-profile-$repeat.log" 2>&1
        echo "PROFILED $mode $repeat"
    done
done
.venv/bin/python -u score_lq_study.py > logs/lq-quality.log 2>&1
echo LQ_STUDY_COMPLETE
