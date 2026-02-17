#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多标签分类训练脚本
使用Hugging Face Transformers进行中文评论的多标签分类训练

支持 backbone 对比：
- 通过 --model_name 指定预训练模型
- 通过 --run_name 指定输出目录名（避免覆盖）

示例：
python train_multilabel.py --model_name hfl/chinese-macbert-base --run_name macbert
python train_multilabel.py --model_name hfl/chinese-roberta-wwm-ext --run_name roberta
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import json
from datetime import datetime

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


def load_dataset(data_path: str = 'data/all_multilabel.csv') -> pd.DataFrame:
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


def prepare_datasets(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
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
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    return train_dataset, val_dataset, test_dataset


def tokenize_function(examples, tokenizer, max_length: int = 150):
    """tokenize文本并准备标签"""
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

    # 使用0.5作为阈值转���为二分类预测
    preds = (probs >= 0.5).astype(int)

    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)

    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)

    metrics = {
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1),
    }

    for i, label_name in enumerate(LABEL_NAMES):
        metrics[f'f1_{label_name}'] = float(per_label_f1[i])

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument("--run_name", type=str, default=None, help="实验名，用于 outputs/<run_name> 输出目录")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("多标签分类训练")
    print("=" * 60)

    model_name = args.model_name
    run_name = args.run_name or model_name.split("/")[-1]

    max_length = args.max_length
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    output_dir = os.path.join("outputs", run_name)
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(output_dir, exist_ok=True)

    # 检查数据集
    data_path = 'data/all_multilabel.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        print("Please run: python build_multilabel_dataset.py")
        return

    # 加载数据
    df = load_dataset(data_path)

    # 准备数据集
    train_dataset, val_dataset, test_dataset = prepare_datasets(df, random_state=args.seed)

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
    columns_to_remove = ['text'] + LABEL_NAMES

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=columns_to_remove
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=columns_to_remove
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=columns_to_remove
    )

    # 设置训练参数
    # transformers==4.57.5 使用 eval_strategy（不是 evaluation_strategy）
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # 如果有GPU则使用fp16
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

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
        try:
            v = float(value)
            print(f"  {key}: {v:.4f}")
        except Exception:
            print(f"  {key}: {value}")

    # 保存指标到 JSON（用于论文取数）
    metrics_path = os.path.join(output_dir, "metrics_test.json")
    payload = {
        "model_name": model_name,
        "run_name": run_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload.update({k: float(v) for k, v in test_results.items() if isinstance(v, (int, float, np.floating))})

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nTest metrics saved to {metrics_path}")

    # 同时写一个对比汇总 CSV（多次运行会追加）
    summary_path = os.path.join("outputs", "backbone_compare.csv")
    row = {"run_name": run_name, "model_name": model_name}
    for k, v in test_results.items():
        if isinstance(v, (int, float, np.floating)):
            row[k] = float(v)

    df_row = pd.DataFrame([row])
    if os.path.exists(summary_path):
        df_row.to_csv(summary_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df_row.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Appended summary to {summary_path}")

    # 保存训练报告（文本版，便于查看）
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Max length: {max_length}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Seed: {args.seed}\n\n")
        f.write("Test Results:\n")
        for key, value in test_results.items():
            try:
                f.write(f"  {key}: {float(value):.4f}\n")
            except Exception:
                f.write(f"  {key}: {value}\n")

    print(f"\nTraining report saved to {report_path}")
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
