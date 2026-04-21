#!/bin/bash

# 补跑缺失气候因子的脚本
# 在screen会话中运行，自动记录所有输出到日志文件

set -e

# 配置
BASE_DIR="/home/leizy/24sum/CFRL"
LOG_DIR="$BASE_DIR/logs/missing_factors_$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/main.log"
ERROR_LOG="$LOG_DIR/errors.log"

# 缺失的气候因子列表
MISSING_FACTORS=(
    "humidity_land-wtd"
    "irradiance-surface_land-wtd" 
    "irradiance-toa_land-wtd"
    "precipitation_land-wtd"
    "temperature_land-wtd"
    "wind-speed_land-wtd"
)

# 创建日志目录
mkdir -p "$LOG_DIR"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

error_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$MAIN_LOG" | tee -a "$ERROR_LOG"
}

# 开始日志记录
log "=== 开始补跑缺失气候因子 ==="
log "日志目录: $LOG_DIR"
log "缺失因子数量: ${#MISSING_FACTORS[@]}"
log "缺失因子列表: ${MISSING_FACTORS[*]}"

# 检查数据目录是否存在
DATA_DIR="$BASE_DIR/data/train_separately_datasets"
if [ ! -d "$DATA_DIR" ]; then
    error_log "数据目录不存在: $DATA_DIR"
    exit 1
fi

# 检查训练脚本是否存在
TRAIN_SCRIPT="$BASE_DIR/src/main_model/scripts/multi_gpu_train_test.sh"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    error_log "训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# 为每个数据类型和步长运行补跑
for DATA_TYPE in "with_prompt" "no_prompt"; do
    for STEP_TYPE in "1" "3" "5"; do
        log "=== 开始处理 $DATA_TYPE ${STEP_TYPE}_step ==="
        
        # 检查该配置的数据是否存在
        STEP_DATA_DIR="$DATA_DIR/$DATA_TYPE/${STEP_TYPE}_step/TH_S"
        if [ ! -d "$STEP_DATA_DIR" ]; then
            error_log "数据目录不存在: $STEP_DATA_DIR"
            continue
        fi
        
        # 检查哪些因子在该配置下缺失
        MISSING_IN_STEP=()
        for factor in "${MISSING_FACTORS[@]}"; do
            factor_dir="$STEP_DATA_DIR/$factor"
            result_dir="$BASE_DIR/data/predicted_results/$DATA_TYPE/TH/${STEP_TYPE}_step/$factor"
            
            # 检查是否真的缺失（没有结果目录或没有metrics.json）
            if [ ! -d "$result_dir" ] || [ ! -f "$result_dir/metrics.json" ]; then
                if [ -d "$factor_dir" ]; then
                    MISSING_IN_STEP+=("$factor")
                fi
            fi
        done
        
        if [ ${#MISSING_IN_STEP[@]} -eq 0 ]; then
            log "$DATA_TYPE ${STEP_TYPE}_step 没有缺失的因子，跳过"
            continue
        fi
        
        log "$DATA_TYPE ${STEP_TYPE}_step 缺失因子: ${MISSING_IN_STEP[*]}"
        
        # 运行训练脚本
        STEP_LOG="$LOG_DIR/${DATA_TYPE}_${STEP_TYPE}_step.log"
        log "运行训练脚本，日志: $STEP_LOG"
        
        if bash "$TRAIN_SCRIPT" "$DATA_TYPE" "$STEP_TYPE" 2>&1 | tee -a "$STEP_LOG"; then
            log "$DATA_TYPE ${STEP_TYPE}_step 完成"
        else
            error_log "$DATA_TYPE ${STEP_TYPE}_step 失败，查看日志: $STEP_LOG"
        fi
        
        # 等待一段时间让GPU冷却
        sleep 30
    done
done

# 生成最终汇总
log "=== 生成最终汇总 ==="
SUMMARY_LOG="$LOG_DIR/final_summary.log"

# 运行汇总脚本
for DATA_TYPE in "with_prompt" "no_prompt"; do
    for STEP_TYPE in "1" "3" "5"; do
        log "生成 $DATA_TYPE ${STEP_TYPE}_step 汇总"
        
        # 生成JSON汇总
        if python3 "$BASE_DIR/src/main_model/scripts/generate_summary.py" \
            --result_dir "$BASE_DIR/data/predicted_results/$DATA_TYPE/TH/${STEP_TYPE}_step" \
            --output_file "$BASE_DIR/data/predicted_results/$DATA_TYPE/TH/${STEP_TYPE}_step_summary.json" \
            2>&1 | tee -a "$SUMMARY_LOG"; then
            log "$DATA_TYPE ${STEP_TYPE}_step JSON汇总完成"
        else
            error_log "$DATA_TYPE ${STEP_TYPE}_step JSON汇总失败"
        fi
        
        # 生成CSV汇总
        if python3 "$BASE_DIR/src/main_model/scripts/generate_summary_csv.py" \
            --data_type "$DATA_TYPE" \
            --step_type "$STEP_TYPE" \
            2>&1 | tee -a "$SUMMARY_LOG"; then
            log "$DATA_TYPE ${STEP_TYPE}_step CSV汇总完成"
        else
            error_log "$DATA_TYPE ${STEP_TYPE}_step CSV汇总失败"
        fi
    done
done

# 计算总时间
log "=== 计算总时间 ==="
if bash "$BASE_DIR/src/main_model/scripts/calculate_all_times.sh" "with_prompt" 2>&1 | tee -a "$SUMMARY_LOG"; then
    log "with_prompt 总时间计算完成"
else
    error_log "with_prompt 总时间计算失败"
fi

if bash "$BASE_DIR/src/main_model/scripts/calculate_all_times.sh" "no_prompt" 2>&1 | tee -a "$SUMMARY_LOG"; then
    log "no_prompt 总时间计算完成"
else
    error_log "no_prompt 总时间计算失败"
fi

# 最终检查
log "=== 最终检查 ==="
FINAL_CHECK_LOG="$LOG_DIR/final_check.log"

for DATA_TYPE in "with_prompt" "no_prompt"; do
    for STEP_TYPE in "1" "3" "5"; do
        echo "=== $DATA_TYPE ${STEP_TYPE}_step 最终状态 ===" | tee -a "$FINAL_CHECK_LOG"
        
        RESULT_DIR="$BASE_DIR/data/predicted_results/$DATA_TYPE/TH/${STEP_TYPE}_step"
        if [ -d "$RESULT_DIR" ]; then
            # 统计完成的因子
            COMPLETED_COUNT=0
            MISSING_COUNT=0
            MISSING_LIST=""
            
            for factor in "${MISSING_FACTORS[@]}"; do
                if [ -f "$RESULT_DIR/$factor/metrics.json" ]; then
                    COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
                else
                    MISSING_COUNT=$((MISSING_COUNT + 1))
                    MISSING_LIST="$MISSING_LIST $factor"
                fi
            done
            
            echo "完成: $COMPLETED_COUNT | 仍缺失: $MISSING_COUNT" | tee -a "$FINAL_CHECK_LOG"
            if [ $MISSING_COUNT -gt 0 ]; then
                echo "仍缺失的因子:$MISSING_LIST" | tee -a "$FINAL_CHECK_LOG"
            fi
        else
            echo "结果目录不存在: $RESULT_DIR" | tee -a "$FINAL_CHECK_LOG"
        fi
    done
done

log "=== 补跑完成 ==="
log "所有日志保存在: $LOG_DIR"
log "主日志: $MAIN_LOG"
log "错误日志: $ERROR_LOG"
log "最终检查: $FINAL_CHECK_LOG"

# 保持screen会话活跃
echo "补跑脚本执行完成，按任意键退出screen会话..."
read -n 1
