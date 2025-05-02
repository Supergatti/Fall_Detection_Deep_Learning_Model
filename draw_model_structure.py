#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 CNN 和 LSTM 模型结构图的独立脚本

该脚本提供了两种方式来绘制模型结构图：
1. 从已有的预训练模型文件加载并绘制
2. 手动构建模型结构并绘制（如果预训练模型无法加载）
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

# 定义保存路径
OUTPUT_DIR = r"C:\Users\38370\Desktop"

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")
    except Exception as e:
        print(f"创建输出目录失败: {e}")
        OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # 回退到脚本所在目录
        print(f"将使用脚本所在目录作为输出目录: {OUTPUT_DIR}")

def load_and_plot_model(model_path, output_file):
    """
    加载预训练模型并绘制其结构图
    
    参数:
    model_path: 模型文件的路径
    output_file: 输出的图片文件名
    """
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return False
    
    try:
        print(f"正在加载模型: {model_path}")
        # 尝试使用自定义加载逻辑
        if "lstm" in model_path.lower():
            model = custom_load_lstm_model(model_path)
        elif "cnn" in model_path.lower():
            model = custom_load_cnn_model(model_path)
        else:
            model = keras.models.load_model(model_path)
        print("模型加载成功！")
        
        # 打印模型摘要
        model.summary()
        
        # 绘制模型结构图，使用完整路径保存
        output_path = os.path.join(OUTPUT_DIR, output_file)
        plot_model(model, show_shapes=True, to_file=output_path)
        print(f"模型结构图已保存为: {output_path}")
        return True
    except Exception as e:
        print(f"加载或绘制模型时出错: {e}")
        print("请尝试使用手动创建的模型结构...")
        return False

def custom_load_lstm_model(model_path):
    """
    自定义加载LSTM模型，处理特殊参数
    """
    try:
        # 创建与保存模型相同结构的模型
        model = create_lstm_model()
        # 尝试只加载权重而不是整个模型结构
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        return model
    except:
        print("尝试备用方式加载LSTM模型...")
        # 创建一个自定义对象标识符来处理额外参数
        custom_objects = {
            'time_major': lambda x: x  # 忽略time_major参数
        }
        try:
            return keras.models.load_model(model_path, custom_objects=custom_objects)
        except Exception as e:
            print(f"备用加载方式也失败: {e}")
            raise e

def custom_load_cnn_model(model_path):
    """
    自定义加载CNN模型，处理输入形状不匹配问题
    """
    try:
        # 创建与保存模型相同结构的模型
        model = create_cnn_model()
        # 尝试只加载权重而不是整个模型结构
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        return model
    except Exception as e:
        print(f"自定义加载CNN模型失败: {e}")
        raise e

def create_lstm_model():
    """
    手动创建 LSTM 模型结构
    """
    # 使用函数式API代替Sequential，以便更好控制
    inputs = keras.layers.Input(shape=(30, 42))
    # 修改LSTM层以匹配预训练模型的参数
    x = keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def create_cnn_model():
    """
    手动创建 CNN 模型结构，修正输入形状问题
    """
    # 根据错误信息调整输入形状为(3, 21, 1)
    inputs = keras.layers.Input(shape=(3, 21, 1))
    
    # 使用Conv2D，但需要确保维度正确
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def manually_create_and_plot_model(model_type, output_file):
    """
    手动创建模型并绘制其结构图
    
    参数:
    model_type: 'lstm' 或 'cnn'
    output_file: 输出的图片文件名
    """
    try:
        if model_type.lower() == 'lstm':
            model = create_lstm_model()
            print("已手动创建 LSTM 模型结构")
        elif model_type.lower() == 'cnn':
            model = create_cnn_model()
            print("已手动创建 CNN 模型结构")
        else:
            print(f"错误: 不支持的模型类型 '{model_type}'")
            return False
        
        # 打印模型摘要
        model.summary()
        
        # 绘制模型结构图，使用完整路径保存
        output_path = os.path.join(OUTPUT_DIR, output_file)
        plot_model(model, show_shapes=True, to_file=output_path)
        print(f"模型结构图已保存为: {output_path}")
        return True
    except Exception as e:
        print(f"创建或绘制模型时出错: {e}")
        return False

def main():
    """主函数"""
    print("开始绘制模型结构图...")
    
    # 设置模型文件路径
    lstm_model_path = os.path.join('Trained_model', 'final_lstm_model.h5')
    cnn_model_path = os.path.join('Trained_model', 'final_cnn_model.h5')
    
    # 尝试从预训练模型加载并绘制
    print("\n=== 尝试加载和绘制 LSTM 模型 ===")
    lstm_success = load_and_plot_model(lstm_model_path, 'lstm_model.png')
    
    print("\n=== 尝试加载和绘制 CNN 模型 ===")
    cnn_success = load_and_plot_model(cnn_model_path, 'cnn_model.png')
    
    # 如果加载失败，尝试手动创建并绘制
    if not lstm_success:
        print("\n预训练 LSTM 模型加载失败，尝试手动创建...")
        manually_create_and_plot_model('lstm', 'lstm_model_manual.png')
    
    if not cnn_success:
        print("\n预训练 CNN 模型加载失败，尝试手动创建...")
        manually_create_and_plot_model('cnn', 'cnn_model_manual.png')
    
    print(f"\n模型结构图绘制完成！所有图片已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()