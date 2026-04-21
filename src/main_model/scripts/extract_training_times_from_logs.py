#!/usr/bin/env python3
"""
从log文件中提取所有模型训练时间并生成详细表格
基于log文件中的"All tasks completed successfully"标记来确认训练完成
"""

import glob
import json
import os
import re
from datetime import datetime

import pandas as pd

EFFICIENCY_PATTERN = re.compile(r"\[EfficiencyMetrics] Epoch (?P<epoch>\d+): (?P<seconds>[0-9.]+)s elapsed(?:, peak GPU memory (?P<memory>[0-9.]+) MB)?")


def format_memory_mb(value):
    """格式化显存峰值（MB）为字符串"""
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if value >= 1024:
        return f"{value / 1024:.2f} GB"
    return f"{value:.2f} MB"


def format_hours(value):
    """将小时数格式化为字符串"""
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    return f"{value:.3f} h"

def parse_log_file(file_path):
    """解析log文件，提取训练时间信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        efficiency_metrics = {
            'has_efficiency_log': False,
            'epochs': [],
            'summary': {},
        }

        json_dir = os.path.dirname(file_path)
        json_path = os.path.join(json_dir, 'efficiency_metrics.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as ef:
                    payload = json.load(ef)
                    efficiency_metrics['has_efficiency_log'] = True
                    efficiency_metrics['summary'] = payload.get('summary', {})
                    efficiency_metrics['epochs'] = payload.get('epochs', [])
            except (json.JSONDecodeError, OSError):
                pass
        else:
            for match in EFFICIENCY_PATTERN.finditer(content):
                efficiency_metrics['has_efficiency_log'] = True
                efficiency_metrics['epochs'].append({
                    'epoch': int(match.group('epoch')) if match.group('epoch') else None,
                    'epoch_runtime_seconds': float(match.group('seconds')),
                    'peak_memory_mb': float(match.group('memory')) if match.group('memory') else None,
                })
        
        # 检查是否完成训练
        if "All tasks completed successfully" not in content:
            return None
            
        # 提取开始时间
        start_patterns = [
            r'\[([^]]+)\] Starting training for',
            r'\[([^]]+)\] Training model for',
            r'\[([^]]+)\] Starting.*training'
        ]
        
        start_time = None
        for pattern in start_patterns:
            match = re.search(pattern, content)
            if match:
                start_time = match.group(1)
                break
        
        if not start_time:
            return None
        
        # 提取完成时间
        completion_patterns = [
            r'\[([^]]+)\] All tasks completed successfully',
            r'\[([^]]+)\] Training completed',
            r'\[([^]]+)\] All tasks completed'
        ]
        
        end_time = None
        for pattern in completion_patterns:
            match = re.search(pattern, content)
            if match:
                end_time = match.group(1)
                break
        
        if not end_time:
            return None
        
        # 解析时间字符串
        try:
            start_dt = datetime.strptime(start_time, '%a %b %d %I:%M:%S %p %Z %Y')
            end_dt = datetime.strptime(end_time, '%a %b %d %I:%M:%S %p %Z %Y')
            
            # 计算总时长
            total_duration = int((end_dt - start_dt).total_seconds())
            
            # 尝试提取训练和测试的具体时间
            training_duration = None
            testing_duration = None
            
            # 查找训练结束时间
            train_end_patterns = [
                r'\[([^]]+)\] Training completed',
                r'\[([^]]+)\] Training finished',
                r'\[([^]]+)\] Model training completed'
            ]
            
            for pattern in train_end_patterns:
                match = re.search(pattern, content)
                if match:
                    train_end_time = match.group(1)
                    try:
                        train_end_dt = datetime.strptime(train_end_time, '%a %b %d %I:%M:%S %p %Z %Y')
                        training_duration = int((train_end_dt - start_dt).total_seconds())
                        testing_duration = total_duration - training_duration
                        break
                    except:
                        continue
            
            return {
                'status': 'completed',
                'start_time': start_time,
                'end_time': end_time,
                'training_duration': training_duration,
                'testing_duration': testing_duration,
                'total_duration': total_duration,
                'efficiency_metrics': efficiency_metrics
            }
            
        except Exception as e:
            print(f"Error parsing time in {file_path}: {e}")
            return {
                'status': 'completed',
                'start_time': start_time,
                'end_time': end_time,
                'training_duration': None,
                'testing_duration': None,
                'total_duration': None,
                'efficiency_metrics': efficiency_metrics
            }
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def format_duration(seconds):
    """将秒数转换为可读格式"""
    if seconds is None:
        return "N/A"
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def extract_all_training_times_from_logs():
    """从log文件中提取所有训练时间信息"""
    base_log_dir = "/home/leizy/24sum/CFRL/logs"
    results = []
    
    # 处理with_prompt和no_prompt
    for data_type in ['with_prompt', 'no_prompt']:
        data_dir = os.path.join(base_log_dir, data_type)
        
        if not os.path.exists(data_dir):
            continue
            
        # 处理每个步骤
        for step_dir in os.listdir(data_dir):
            if not step_dir.startswith('T5_') or not os.path.isdir(os.path.join(data_dir, step_dir)):
                continue
                
            step_path = os.path.join(data_dir, step_dir)
            step_name = step_dir.replace('T5_', '').replace('_step', '')
            
            # 处理每个log文件
            log_files = glob.glob(os.path.join(step_path, "*.log"))
            
            for log_file in log_files:
                # 从文件名提取气候因子和GPU信息
                filename = os.path.basename(log_file)
                # 格式: climate_factor_gpuX.log
                name_part = filename.replace('.log', '')
                parts = name_part.rsplit('_', 1)  # 分离GPU信息
                
                if len(parts) == 2:
                    climate_factor = parts[0]
                    gpu_info = parts[1]
                else:
                    climate_factor = name_part
                    gpu_info = "unknown"
                
                # 解析log文件
                time_info = parse_log_file(log_file)
                if time_info:
                    entry = {
                        'data_type': data_type,
                        'step': step_name,
                        'climate_factor': climate_factor,
                        'gpu': gpu_info,
                        'status': time_info['status'],
                        'start_time': time_info['start_time'],
                        'end_time': time_info['end_time'],
                        'training_duration_seconds': time_info['training_duration'],
                        'testing_duration_seconds': time_info['testing_duration'],
                        'total_duration_seconds': time_info['total_duration'],
                        'training_duration_formatted': format_duration(time_info['training_duration']),
                        'testing_duration_formatted': format_duration(time_info['testing_duration']),
                        'total_duration_formatted': format_duration(time_info['total_duration'])
                    }

                    efficiency = time_info.get('efficiency_metrics') or {}
                    summary = efficiency.get('summary') or {}
                    epochs = efficiency.get('epochs') or []
                    entry['has_efficiency_log'] = efficiency.get('has_efficiency_log', False)
                    entry['gpu_hours'] = summary.get('gpu_hours')
                    entry['train_runtime_hours'] = summary.get('train_runtime_hours')
                    entry['peak_memory_mb'] = summary.get('max_peak_memory_mb') or max((epoch.get('peak_memory_mb') for epoch in epochs if epoch.get('peak_memory_mb')), default=None)
                    entry['peak_memory_formatted'] = format_memory_mb(entry['peak_memory_mb'])
                    entry['epochs_logged'] = len(epochs)

                    results.append(entry)
    
    return results

def generate_markdown_table(results):
    """生成Markdown表格"""
    if not results:
        return "# 训练时间与效率指标统计\n\n没有找到任何训练时间数据。\n"
    
    # 按data_type, step, climate_factor排序
    results.sort(key=lambda x: (x['data_type'], x['step'], x['climate_factor']))
    
    markdown = "# 96个模型训练时间与效率指标统计 (基于Log文件分析)\n\n"
    markdown += f"**统计时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # 总体统计
    total_models = len(results)
    completed_models = len([r for r in results if r['status'] == 'completed'])
    
    markdown += "## 总体统计\n\n"
    markdown += f"- **总模型数**: {total_models}\n"
    markdown += f"- **成功完成**: {completed_models}\n"
    markdown += f"- **成功率**: {completed_models/total_models*100:.1f}%\n\n"
    
    # 按数据类型分组
    for data_type in ['with_prompt', 'no_prompt']:
        type_results = [r for r in results if r['data_type'] == data_type]
        if not type_results:
            continue
            
        markdown += f"## {data_type.upper()} 数据集\n\n"
        
        # 按步骤分组
        for step in ['1', '3', '5']:
            step_results = [r for r in type_results if r['step'] == step]
            if not step_results:
                continue
                
            markdown += f"### {step}步预测\n\n"
            
            # 创建表格
            markdown += "| 气候因子 | GPU | 状态 | 开始时间 | 结束时间 | 训练时长 | 总时长 | GPU小时 | 显存峰值 |\n"
            markdown += "|---------|-----|------|----------|----------|----------|--------|---------|----------|\n"
            
            for result in step_results:
                status_emoji = "✅" if result['status'] == 'completed' else "❌"
                gpu_hours_str = format_hours(result.get('gpu_hours')) if result.get('gpu_hours') else "N/A"
                peak_memory_str = result.get('peak_memory_formatted') or "N/A"
                markdown += (
                    f"| {result['climate_factor']} | {result['gpu']} | {status_emoji} | {result['start_time']} | {result['end_time']} | "
                    f"{result['training_duration_formatted']} | {result['total_duration_formatted']} | {gpu_hours_str} | {peak_memory_str} |\n"
                )
            
            # 步骤统计
            step_completed = len([r for r in step_results if r['status'] == 'completed'])
            step_total = len(step_results)
            markdown += f"\n**{step}步统计**: {step_completed}/{step_total} 完成 ({step_completed/step_total*100:.1f}%)\n\n"
        
        # 数据类型统计
        type_completed = len([r for r in type_results if r['status'] == 'completed'])
        type_total = len(type_results)
        markdown += f"**{data_type}总计**: {type_completed}/{type_total} 完成 ({type_completed/type_total*100:.1f}%)\n\n"
    
    # 详细统计信息
    markdown += "## 详细统计信息\n\n"
    
    # 按状态分组统计
    completed_results = [r for r in results if r['status'] == 'completed']
    if completed_results:
        # 训练时间统计
        training_times = [r['training_duration_seconds'] for r in completed_results if r['training_duration_seconds']]
        if training_times:
            avg_training = sum(training_times) / len(training_times)
            min_training = min(training_times)
            max_training = max(training_times)
            
            markdown += f"### 训练时间统计 (已完成模型)\n"
            markdown += f"- **平均训练时间**: {format_duration(int(avg_training))}\n"
            markdown += f"- **最短训练时间**: {format_duration(min_training)}\n"
            markdown += f"- **最长训练时间**: {format_duration(max_training)}\n\n"
        
        # 测试时间统计
        testing_times = [r['testing_duration_seconds'] for r in completed_results if r['testing_duration_seconds']]
        if testing_times:
            avg_testing = sum(testing_times) / len(testing_times)
            min_testing = min(testing_times)
            max_testing = max(testing_times)
            
            markdown += f"### 测试时间统计 (已完成模型)\n"
            markdown += f"- **平均测试时间**: {format_duration(int(avg_testing))}\n"
            markdown += f"- **最短测试时间**: {format_duration(min_testing)}\n"
            markdown += f"- **最长测试时间**: {format_duration(max_testing)}\n\n"
        
        # 总时间统计
        total_times = [r['total_duration_seconds'] for r in completed_results if r['total_duration_seconds']]
        if total_times:
            avg_total = sum(total_times) / len(total_times)
            min_total = min(total_times)
            max_total = max(total_times)
            
            markdown += f"### 总时间统计 (已完成模型)\n"
            markdown += f"- **平均总时间**: {format_duration(int(avg_total))}\n"
            markdown += f"- **最短总时间**: {format_duration(min_total)}\n"
            markdown += f"- **最长总时间**: {format_duration(max_total)}\n\n"

        # 效率指标统计
        gpu_hours_list = [r['gpu_hours'] for r in completed_results if r.get('gpu_hours')]
        peak_memory_list = [r['peak_memory_mb'] for r in completed_results if r.get('peak_memory_mb')]

        if gpu_hours_list:
            markdown += "### GPU 小时统计 (已完成模型)\n"
            avg_gpu_hours = sum(gpu_hours_list) / len(gpu_hours_list)
            markdown += f"- **平均总 GPU 小时**: {avg_gpu_hours:.3f} h\n"
            markdown += f"- **最少**: {min(gpu_hours_list):.3f} h\n"
            markdown += f"- **最多**: {max(gpu_hours_list):.3f} h\n\n"

        if peak_memory_list:
            markdown += "### 显存峰值统计 (已完成模型)\n"
            avg_peak = sum(peak_memory_list) / len(peak_memory_list)
            markdown += f"- **平均峰值**: {format_memory_mb(avg_peak)}\n"
            markdown += f"- **最小峰值**: {format_memory_mb(min(peak_memory_list))}\n"
            markdown += f"- **最大峰值**: {format_memory_mb(max(peak_memory_list))}\n\n"
    
    return markdown

def main():
    """主函数"""
    print("开始从log文件中提取训练时间信息...")
    
    # 提取所有时间信息
    results = extract_all_training_times_from_logs()
    print(f"找到 {len(results)} 个完成训练的模型")
    
    # 生成Markdown表格
    markdown_content = generate_markdown_table(results)
    
    # 保存到文件
    output_file = "/home/leizy/24sum/CFRL/logs/training_times_from_logs_summary.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"训练时间统计已保存到: {output_file}")
    
    # 同时保存CSV格式
    if results:
        df = pd.DataFrame(results)
        csv_file = "/home/leizy/24sum/CFRL/logs/training_times_from_logs_summary.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"CSV格式已保存到: {csv_file}")
        
        # 打印简要统计
        print(f"\n=== 简要统计 ===")
        print(f"总模型数: {len(results)}")
        print(f"with_prompt: {len([r for r in results if r['data_type'] == 'with_prompt'])}")
        print(f"no_prompt: {len([r for r in results if r['data_type'] == 'no_prompt'])}")
        
        for data_type in ['with_prompt', 'no_prompt']:
            type_results = [r for r in results if r['data_type'] == data_type]
            print(f"\n{data_type}:")
            for step in ['1', '3', '5']:
                step_count = len([r for r in type_results if r['step'] == step])
                print(f"  {step}步: {step_count}个模型")

if __name__ == "__main__":
    main()

