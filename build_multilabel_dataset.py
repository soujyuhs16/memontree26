#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建多标签数据集脚本
从原始数据集中合并6个CSV文件，生成用于多标签分类的训练数据
"""

import os
import pandas as pd
import numpy as np


def load_and_process_csv(filepath, label_name):
    """
    加载并处理单个CSV文件
    
    Args:
        filepath: CSV文件路径
        label_name: 标签名称 (porn, abuse, region, gender, race, occupation)
    
    Returns:
        DataFrame with columns: text, label_name (0/1)
    """
    print(f"Processing {filepath}...")
    
    # 读取CSV，处理UTF-8 BOM
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame(columns=['text', label_name])
    
    print(f"  Initial rows: {len(df)}")
    
    # 检查必需的列
    if 'Sentence' not in df.columns or 'Type' not in df.columns:
        print(f"  Warning: Missing required columns in {filepath}")
        return pd.DataFrame(columns=['text', label_name])
    
    # 重命名列
    df = df.rename(columns={'Sentence': 'text'})
    
    # 只保留需要的列
    df = df[['text', 'Type']].copy()
    
    # 删除空值
    df = df.dropna(subset=['text', 'Type'])
    
    # 去除空白
    df['text'] = df['text'].astype(str).str.strip()
    
    # 过滤空文本和"nan"字符串
    df = df[df['text'] != '']
    df = df[df['text'].str.lower() != 'nan']
    
    print(f"  After cleaning: {len(df)} rows")
    
    # 将Type转换为0/1标签
    df[label_name] = (df['Type'] == 'Harmful').astype(int)
    
    # 只保留text和标签列
    df = df[['text', label_name]]
    
    print(f"  Harmful samples: {df[label_name].sum()}")
    
    return df


def main():
    """主函数：构建多标签数据集"""
    
    # 定义输入文件和对应的标签
    file_configs = [
        ('Dataset/SexHarmset/SexHarmSet.csv', 'porn'),
        ('Dataset/AbuseSet/AbuseSet.csv', 'abuse'),
        ('Dataset/BiasSet/Bias_region.csv', 'region'),
        ('Dataset/BiasSet/BiasSet_genden.csv', 'gender'),
        ('Dataset/BiasSet/Bias_race.csv', 'race'),
        ('Dataset/BiasSet/Bias_occupation.csv', 'occupation'),
    ]
    
    print("=" * 60)
    print("开始构建多标签数据集")
    print("=" * 60)
    
    # 加载并处理所有文件
    dfs = []
    for filepath, label_name in file_configs:
        df = load_and_process_csv(filepath, label_name)
        if len(df) > 0:
            dfs.append(df)
    
    if not dfs:
        print("Error: No valid data loaded!")
        return
    
    print("\n" + "=" * 60)
    print("合并数据集")
    print("=" * 60)
    
    # 合并所有数据框
    # 使用outer join来保留所有文本，未出现的标签填充0
    combined_df = dfs[0]
    for df in dfs[1:]:
        combined_df = pd.merge(combined_df, df, on='text', how='outer')
    
    # 填充缺失的标签值为0
    label_columns = ['porn', 'abuse', 'region', 'gender', 'race', 'occupation']
    for col in label_columns:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].fillna(0).astype(int)
    
    print(f"合并后总行数: {len(combined_df)}")
    
    # 按text分组，使用max聚合（如果同一文本在多个数据集中出现，取最大值）
    print("\n按text分组并聚合...")
    grouped_df = combined_df.groupby('text', as_index=False).max()
    
    print(f"聚合后行数: {len(grouped_df)}")
    
    # 确保列的顺序
    final_columns = ['text'] + label_columns
    grouped_df = grouped_df[final_columns]
    
    # 创建输出目录
    os.makedirs('data', exist_ok=True)
    
    # 保存到CSV
    output_path = 'data/all_multilabel.csv'
    grouped_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("数据集统计信息")
    print("=" * 60)
    print(f"输出文件: {output_path}")
    print(f"总行数: {len(grouped_df)}")
    print("\n各标签正样本数:")
    for col in label_columns:
        positive_count = grouped_df[col].sum()
        print(f"  {col}: {positive_count} ({positive_count/len(grouped_df)*100:.2f}%)")
    
    print("\nNull值统计:")
    null_counts = grouped_df.isnull().sum()
    if null_counts.sum() == 0:
        print("  无null值")
    else:
        print(null_counts)
    
    print("\n数据集构建完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
