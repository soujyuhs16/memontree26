#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多标签分类训练脚本
使用Hugging Face Transformers进行中文评论的多标签分类训练
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)


# 标签列表
LABEL_NAMES = ['porn', 'abuse', 'region', 'gender', 'race', 'occupation']
NUM_LABELS = len(LABEL_NAMES)


def load_dataset(data_path='data/all_multilabel.csv'):
    """加载数据集"""
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    print(f"Loaded {len(df)} samples")
    
    # 检查数据
    print("\nLabel distribution:")
    for label in LABEL_NAMES:
        count = df[label].sum()
        print(f"  {label}: {count} ({count/len(df)*100:.2f}%)")
    
    return df


def prepare_datasets(df, test_size=0.2, val_size=0.1, random_state=42):
    """准备训练、验证和测试数据集"""
    # 分割为训练+验证 和 测试集
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # 再将训练+验证分割为训练和验证
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # 转换为Hugging Face Dataset格式
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, val_dataset, test_dataset


def tokenize_function(examples, tokenizer, max_length=150):
    """tokenize文本并准备标签"""
    # Tokenize文本
    tokenized = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
    
    # 准备多标签（需要是float类型用于BCEWithLogitsLoss）
    labels = []
    for i in range(len(examples['text'])):
        label_vector = [float(examples[label][i]) for label in LABEL_NAMES]
        labels.append(label_vector)
    
    tokenized['labels'] = labels
    return tokenized


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    
    # 对predictions应用sigmoid得到概率
    probs = 1 / (1 + np.exp(-predictions))
    
    # 使用0.5作为阈值转换为二分类预测
    preds = (probs >= 0.5).astype(int)
    
    # 计算各种F1分数
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    # 计算每个标签的F1分数
    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    
    metrics = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
    }
    
    # 添加每个标签的F1分数
    for i, label_name in enumerate(LABEL_NAMES):
        metrics[f'f1_{label_name}'] = per_label_f1[i]
    
    return metrics


def main():
    """主训练函数"""
    print("=" * 60)
    print("多标签分类训练")
    print("=" * 60)
    
    # 配置
    model_name = 'hfl/chinese-macbert-base'
    max_length = 150
    batch_size = 16
    num_epochs = 3
    learning_rate = 2e-5
    output_dir = 'outputs'
    best_model_dir = 'outputs/best_model'
    
    # 检查数据集
    if not os.path.exists('data/all_multilabel.csv'):
        print("Error: data/all_multilabel.csv not found!")
        print("Please run: python build_multilabel_dataset.py")
        return
    
    # 加载数据
    df = load_dataset()
    
    # 准备数据集
    train_dataset, val_dataset, test_dataset = prepare_datasets(df)
    
    # 加载tokenizer和模型
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )
    
    # Tokenize数据集
    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=['text'] + LABEL_NAMES
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=['text'] + LABEL_NAMES
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=['text'] + LABEL_NAMES
    )
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # 如果有GPU则使用fp16
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 开始训练
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    train_result = trainer.train()
    
    # 保存最佳模型
    print(f"\nSaving best model to {best_model_dir}")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    
    # 在测试集上评估
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    test_results = trainer.evaluate(test_dataset)
    
    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # 保存训练报告
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Max length: {max_length}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n\n")
        f.write("Test Results:\n")
        for key, value in test_results.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\nTraining report saved to {report_path}")
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
