#!/bin/bash
# ============================================================
# SparseBEV 鲁棒性基准测试 — 一键运行脚本
# 
# 测试内容:
#   1. Clean 基准测试
#   2. 丢帧测试: 10%-90% (discrete)
#   3. 外参扰动测试: L1-L4 × single/all
#   4. 遮挡测试: S1-S4 (exp=1.0/2.0/3.0/5.0)
#
# 用法:
#   bash run_all_robust_tests.sh [WEIGHTS] [CONFIG] [GPUS]
#
# 示例:
#   bash run_all_robust_tests.sh ckpts/r50_nuimg_704x256.pth configs/r50_nuimg_704x256.py 4
# ============================================================

set -e

# ---- 参数配置 ----
WEIGHTS=${1:-"ckpts/r50_nuimg_704x256.pth"}
CONFIG=${2:-"configs/r50_nuimg_704x256.py"}
GPUS=${3:-4}

# 数据路径
NOISE_PKL="data/nuscenes/nuscenes_infos_val_with_noise.pkl"
DROP_PKL="data/nuscenes/nuscenes_infos_val_with_noise_Drop.pkl"
MASK_DIR="robust_benchmark/Mud_Mask_selected"
OUTPUT_DIR="robust_results"

# 分布式配置
PORT=${PORT:-29500}

mkdir -p ${OUTPUT_DIR}

# ---- 辅助函数 ----
run_test() {
    local desc="$1"
    shift
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${desc}"
    echo "============================================================"
    
    if [ ${GPUS} -gt 1 ]; then
        python -m torch.distributed.launch \
            --nproc_per_node=${GPUS} \
            --master_port=${PORT} \
            robust_val.py \
            --config ${CONFIG} \
            --weights ${WEIGHTS} \
            --world_size ${GPUS} \
            --output-dir ${OUTPUT_DIR} \
            "$@"
    else
        python robust_val.py \
            --config ${CONFIG} \
            --weights ${WEIGHTS} \
            --output-dir ${OUTPUT_DIR} \
            "$@"
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ ${desc} 完成"
    
    # 每次测试后递增端口，避免冲突
    PORT=$((PORT + 1))
}

# ============================================================
# 1. Clean 基准测试
# ============================================================
run_test "Clean 基准测试" \
    --noise-type clean

# ============================================================
# 2. 丢帧测试 (10% - 90%, discrete)
# ============================================================
for ratio in 10 20 30 40 50 60 70 80 90; do
    run_test "丢帧测试 ratio=${ratio}%" \
        --noise-type drop \
        --drop-ratio ${ratio} \
        --noise-pkl "${DROP_PKL}"
done

# ============================================================
# 3. 外参扰动测试 (L1-L4 × single/all)
# ============================================================
for level in L1 L2 L3 L4; do
    for ntype in single all; do
        run_test "外参扰动测试 ${level}/${ntype}" \
            --noise-type extrinsics \
            --extrinsics-level ${level} \
            --extrinsics-type ${ntype} \
            --noise-pkl "${NOISE_PKL}"
    done
done

# ============================================================
# 4. 遮挡测试 (S1-S4: exp=1.0, 2.0, 3.0, 5.0)
# ============================================================
for exp in 1.0 2.0 3.0 5.0; do
    run_test "遮挡测试 exp=${exp}" \
        --noise-type occlusion \
        --occlusion-exp ${exp} \
        --noise-pkl "${NOISE_PKL}" \
        --mask-dir "${MASK_DIR}"
done

# ============================================================
# 5. 汇总结果并计算 RDRR
# ============================================================
echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有测试完成，计算 RDRR..."
echo "============================================================"

python compute_rdrr.py --results-dir ${OUTPUT_DIR}

echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部完成！结果已保存到 ${OUTPUT_DIR}/"
echo "============================================================"
