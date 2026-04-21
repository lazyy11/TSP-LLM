#!/usr/bin/env python3
"""
从log文件中准确提取训练时间和测试时间
基于train_runtime和测试进度条的时间信息
"""

import os
import re
from datetime import datetime
import pandas as pd
import glob

def parse_time_string(time_str):
    """解析时间字符串，转换为秒数"""
    if not time_str:
        return None
    
    # 处理格式如 "6:45:06.99" 或 "19:31"
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 3:  # HH:MM:SS.ss
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return int(hours * 3600 + minutes * 60 + seconds)
        elif len(parts) == 2:  # MM:SS
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
    
    return None

def extract_training_time(log_content):
    """从log内容中提取训练时间"""
    # 查找train_runtime
    patterns = [
        r'train_runtime\s*=\s*([0-9:\.]+)',
        r"'train_runtime':\s*([0-9\.]+)",
        r'"train_runtime":\s*([0-9\.]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, log_content)
        if match:
            time_str = match.group(1)
            # 如果是纯数字（秒数），直接返回
            if time_str.replace('.', '').isdigit():
                return int(float(time_str))
            # 如果是时间格式，解析
            return parse_time_string(time_str)
    
    return None

def extract_testing_time(log_content):
    """从log内容中提取测试时间"""
    # 查找测试进度条，格式如 "100%|██████████| 4711/4711 [19:31<00:00, 4.02it/s]"
    # 我们需要找到最后一个有实际时间的进度条（不是00:00<00:00的）
    patterns = [
        r'100%.*?\[([0-9:]+)<00:00',
        r'100%.*?\[([0-9:]+)<00:00'
    ]
    
    # 找到所有匹配的进度条
    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, log_content)
        all_matches.extend(matches)
    
    # 过滤掉00:00的情况，返回最后一个有效的时间
    valid_times = []
    for time_str in all_matches:
        if time_str != '00:00':
            valid_times.append(time_str)
    
    if valid_times:
        # 返回最后一个有效时间（通常是测试时间）
        return parse_time_string(valid_times[-1])
    
    return None

def extract_start_end_times(log_content):
    """提取开始和结束时间"""
    # 开始时间
    start_patterns = [
        r'\[([^]]+)\] Starting training for',
        r'\[([^]]+)\] Training model for'
    ]
    
    start_time = None
    for pattern in start_patterns:
        match = re.search(pattern, log_content)
        if match:
            start_time = match.group(1)
            break
    
    # 结束时间
    end_patterns = [
        r'\[([^]]+)\] All tasks completed successfully',
        r'\[([^]]+)\] Training completed successfully'
    ]
    
    end_time = None
    for pattern in end_patterns:
        match = re.search(pattern, log_content)
        if match:
            end_time = match.group(1)
            break
    
    return start_time, end_time

def parse_log_file(file_path):
    """解析log文件，提取训练时间信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否完成训练
        if "All tasks completed successfully" not in content:
            return None
        
        # 提取训练时间
        training_duration = extract_training_time(content)
        
        # 提取测试时间
        testing_duration = extract_testing_time(content)
        
        # 提取开始和结束时间
        start_time, end_time = extract_start_end_times(content)
        
        # 计算总时间
        total_duration = None
        if training_duration and testing_duration:
            total_duration = training_duration + testing_duration
        
        return {
            'status': 'completed',
            'start_time': start_time,
            'end_time': end_time,
            'training_duration_seconds': training_duration,
            'testing_duration_seconds': testing_duration,
            'total_duration_seconds': total_duration
        }
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
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

def extract_all_accurate_training_times():
    """从log文件中提取所有准确的训练时间信息"""
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
                    results.append({
                        'data_type': data_type,
                        'step': step_name,
                        'climate_factor': climate_factor,
                        'gpu': gpu_info,
                        'status': time_info['status'],
                        'start_time': time_info['start_time'],
                        'end_time': time_info['end_time'],
                        'training_duration_seconds': time_info['training_duration_seconds'],
                        'testing_duration_seconds': time_info['testing_duration_seconds'],
                        'total_duration_seconds': time_info['total_duration_seconds'],
                        'training_duration_formatted': format_duration(time_info['training_duration_seconds']),
                        'testing_duration_formatted': format_duration(time_info['testing_duration_seconds']),
                        'total_duration_formatted': format_duration(time_info['total_duration_seconds'])
                    })
    
    return results

def generate_markdown_table(results):
    """生成Markdown表格"""
    if not results:
        return "# 训练时间统计\n\n没有找到任何训练时间数据。\n"
    
    # 按data_type, step, climate_factor排序
    results.sort(key=lambda x: (x['data_type'], x['step'], x['climate_factor']))
    
    markdown = "# 96个模型训练时间详细统计 (基于Log文件准确提取)\n\n"
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
            markdown += "| 气候因子 | GPU | 状态 | 开始时间 | 结束时间 | 训练时长 | 测试时长 | 总时长 |\n"
            markdown += "|---------|-----|------|----------|----------|----------|----------|--------|\n"
            
            for result in step_results:
                status_emoji = "✅" if result['status'] == 'completed' else "❌"
                markdown += f"| {result['climate_factor']} | {result['gpu']} | {status_emoji} | {result['start_time']} | {result['end_time']} | {result['training_duration_formatted']} | {result['testing_duration_formatted']} | {result['total_duration_formatted']} |\n"
            
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
    
    return markdown

def main():
    """主函数"""
    print("开始从log文件中准确提取训练时间信息...")
    
    # 提取所有时间信息
    results = extract_all_accurate_training_times()
    print(f"找到 {len(results)} 个完成训练的模型")
    
    # 生成Markdown表格
    markdown_content = generate_markdown_table(results)
    
    # 保存到文件
    output_file = "/home/leizy/24sum/CFRL/logs/accurate_training_times_summary.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"准确训练时间统计已保存到: {output_file}")
    
    # 同时保存CSV格式
    if results:
        df = pd.DataFrame(results)
        csv_file = "/home/leizy/24sum/CFRL/logs/accurate_training_times_summary.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"CSV格式已保存到: {csv_file}")
        
        # 打印简要统计
        print(f"\n=== 简要统计 ===")
        print(f"总模型数: {len(results)}")
        print(f"with_prompt: {len([r for r in results if r['data_type'] == 'with_prompt'])}")
        print(f"no_prompt: {len([r for r in results if r['data_type'] == 'no_prompt'])}")
        
        # 统计有训练时间信息的模型
        training_time_count = len([r for r in results if r['training_duration_seconds']])
        testing_time_count = len([r for r in results if r['testing_duration_seconds']])
        print(f"有训练时间信息的模型: {training_time_count}")
        print(f"有测试时间信息的模型: {testing_time_count}")
        
        for data_type in ['with_prompt', 'no_prompt']:
            type_results = [r for r in results if r['data_type'] == data_type]
            print(f"\n{data_type}:")
            for step in ['1', '3', '5']:
                step_count = len([r for r in type_results if r['step'] == step])
                print(f"  {step}步: {step_count}个模型")

if __name__ == "__main__":
    main()

