#!/usr/bin/env python3
"""
将 agent_history 目录下的 JSON 文件中的截图数据转换为 GIF 动画
"""

import json
import base64
import os
import glob
from PIL import Image
import io
from datetime import datetime

def extract_screenshots_from_json(json_file_path):
    """从 JSON 文件中提取截图数据"""
    screenshots = []
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否有 history 字段
        if 'history' in data:
            for entry in data['history']:
                if 'state' in entry and 'screenshot' in entry['state']:
                    screenshot_data = entry['state']['screenshot']
                    if screenshot_data and screenshot_data.strip():
                        screenshots.append(screenshot_data)
        
        # 如果没有 history 字段，检查是否直接有 screenshot 字段
        elif 'screenshot' in data:
            screenshot_data = data['screenshot']
            if screenshot_data and screenshot_data.strip():
                screenshots.append(screenshot_data)
        
        return screenshots
    
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"处理文件 {json_file_path} 时出错: {e}")
        return []

def base64_to_image(base64_string):
    """将 base64 字符串转换为 PIL Image 对象"""
    try:
        # 移除可能的数据URI前缀
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',', 1)[1]
        
        # 解码 base64
        image_data = base64.b64decode(base64_string)
        
        # 创建 PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为 RGB 模式（GIF 需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    
    except Exception as e:
        print(f"转换 base64 图片时出错: {e}")
        return None

def process_agent_history_directory(input_dir, output_dir="gif_output"):
    """处理 agent_history 目录下的所有文件"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    processed_count = 0
    
    for json_file in json_files:
        print(f"处理文件: {os.path.basename(json_file)}")
        
        # 提取截图
        screenshots = extract_screenshots_from_json(json_file)
        
        if not screenshots:
            print(f"  跳过: 没有找到截图数据")
            continue
        
        # 转换截图为图片
        images = []
        for i, screenshot_b64 in enumerate(screenshots):
            image = base64_to_image(screenshot_b64)
            if image:
                images.append(image)
            else:
                print(f"  警告: 第 {i+1} 张截图转换失败")
        
        if not images:
            print(f"  跳过: 没有有效的图片")
            continue
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.gif")
        
        try:
            # 创建 GIF
            if len(images) == 1:
                # 只有一张图片，保存为静态图片
                images[0].save(output_file.replace('.gif', '.png'))
                print(f"  保存为静态图片: {output_file.replace('.gif', '.png')}")
            else:
                # 多张图片，创建动画 GIF
                images[0].save(
                    output_file,
                    save_all=True,
                    append_images=images[1:],
                    duration=1000,  # 每帧显示 1 秒
                    loop=0  # 无限循环
                )
                print(f"  保存为 GIF: {output_file} (包含 {len(images)} 帧)")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  保存文件时出错: {e}")
    
    print(f"\n处理完成! 成功处理了 {processed_count} 个文件")
    print(f"输出目录: {output_dir}")

def create_combined_gif(input_dir, output_file="combined_agent_history.gif"):
    """将所有文件的截图合并成一个大的 GIF"""
    
    print("创建合并的 GIF...")
    
    # 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    # 按文件名排序
    json_files.sort()
    
    all_images = []
    
    for json_file in json_files:
        print(f"从 {os.path.basename(json_file)} 提取截图...")
        
        screenshots = extract_screenshots_from_json(json_file)
        
        for screenshot_b64 in screenshots:
            image = base64_to_image(screenshot_b64)
            if image:
                all_images.append(image)
    
    if not all_images:
        print("没有找到任何有效的截图")
        return
    
    print(f"总共找到 {len(all_images)} 张截图")
    
    try:
        # 创建合并的 GIF
        all_images[0].save(
            output_file,
            save_all=True,
            append_images=all_images[1:],
            duration=500,  # 每帧显示 0.5 秒
            loop=0  # 无限循环
        )
        print(f"合并的 GIF 已保存: {output_file}")
    
    except Exception as e:
        print(f"创建合并 GIF 时出错: {e}")

if __name__ == "__main__":
    # 设置路径
    input_directory = "tmp/agent_history"
    
    print("Agent History 转 GIF 工具")
    print("=" * 50)
    
    # 检查输入目录
    if not os.path.exists(input_directory):
        print(f"错误: 找不到目录 {input_directory}")
        exit(1)
    
    # 处理选项
    print("选择处理方式:")
    print("1. 为每个文件单独创建 GIF")
    print("2. 创建一个包含所有截图的合并 GIF") 
    print("3. 两种方式都执行")
    
    choice = input("请选择 (1/2/3): ").strip()
    
    if choice == "1" or choice == "3":
        process_agent_history_directory(input_directory)
    
    if choice == "2" or choice == "3":
        create_combined_gif(input_directory)
    
    if choice not in ["1", "2", "3"]:
        print("无效选择，默认执行两种方式")
        process_agent_history_directory(input_directory)
        create_combined_gif(input_directory) 