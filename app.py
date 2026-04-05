#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit演示应用
多标签智能审核系统的交互式界面
"""

import os
import re
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 标签名称
LABEL_NAMES = ['porn', 'abuse', 'region', 'gender', 'race', 'occupation']
LABEL_NAMES_CN = {
    'porn': '色情',
    'abuse': '辱骂',
    'region': '地域歧视',
    'gender': '性别歧视',
    'race': '种族歧视',
    'occupation': '职业歧视'
}


def list_available_best_models(outputs_dir: str = "outputs") -> dict:
    """
    扫描 outputs 目录下所有包含 best_model 的子目录。
    返回：display_name -> model_path
    例如： 'roberta/best_model' -> 'outputs/roberta/best_model'
    """
    candidates = {}

    if not os.path.isdir(outputs_dir):
        return candidates

    # 兼容旧路径 outputs/best_model（如果存在）
    legacy = os.path.join(outputs_dir, "best_model")
    if os.path.isdir(legacy):
        candidates["best_model"] = legacy

    for run_name in sorted(os.listdir(outputs_dir)):
        run_dir = os.path.join(outputs_dir, run_name)
        if not os.path.isdir(run_dir):
            continue

        best_dir = os.path.join(run_dir, "best_model")
        if os.path.isdir(best_dir):
            display = f"{run_name}/best_model"
            candidates[display] = best_dir

    return candidates


@st.cache_resource
def load_model(model_path: str):
    """加载训练好的模型（按 model_path 缓存）"""
    if not os.path.exists(model_path):
        st.error(f"模型未找到: {model_path}")
        st.info("请先运行训练脚本: python train_multilabel.py")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model


def apply_rules(text):
    """应用简单规则检测"""
    matched_rules = []
    rule_scores = {label: 0.0 for label in LABEL_NAMES}

    # URL检测
    url_pattern = r'https?://[^\s]+'
    if re.search(url_pattern, text):
        matched_rules.append("检测到URL链接")
        rule_scores['abuse'] += 0.1  # 可能是垃圾信息

    # 手机号检测
    phone_pattern = r'1[3-9]\d{9}'
    if re.search(phone_pattern, text):
        matched_rules.append("检测到手机号")
        rule_scores['abuse'] += 0.2  # 可能是广告/垃圾信息

    # 重复字符检测（灌水）
    repeat_pattern = r'(.)\1{4,}'  # 5个或更多相同字符
    if re.search(repeat_pattern, text):
        matched_rules.append("检测到重复字符（可能灌水）")
        rule_scores['abuse'] += 0.15

    # 短文本大量重复字符
    if len(text) < 20 and len(set(text)) < 5:
        matched_rules.append("文本过于简单/重复")
        rule_scores['abuse'] += 0.1

    # 检测色情关键词（简单示例）
    porn_keywords = ['做爱', '性交', '色情', '黄色', '裸体', '性爱', '淫']
    for keyword in porn_keywords:
        if keyword in text:
            matched_rules.append(f"检测到敏感词: {keyword}")
            rule_scores['porn'] += 0.3
            break

    # 检测辱骂关键词
    abuse_keywords = ['傻逼', '去死', '垃圾', '废物', '蠢货', '白痴']
    for keyword in abuse_keywords:
        if keyword in text:
            matched_rules.append(f"检测到辱骂词: {keyword}")
            rule_scores['abuse'] += 0.3
            break

    return matched_rules, rule_scores


def predict_single(text, tokenizer, model, thresholds):
    """对单条文本进行预测"""
    # 应用规则
    matched_rules, rule_scores = apply_rules(text)

    # 模型预测
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=150)

    # 确保 inputs 和 model 在同一 device（GPU/CPU）
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs_arr = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

    # 转成 dict（方便后续 UI 使用 probs[label]）
    probs = {label: float(probs_arr[i]) for i, label in enumerate(LABEL_NAMES)}

    # 融合规则分数（简单加权）
    fused_probs = {}
    decisions = {}

    for label in LABEL_NAMES:
        base_prob = probs[label]
        rule_boost = rule_scores[label]
        fused_prob = min(1.0, base_prob + rule_boost)
        fused_probs[label] = fused_prob

        # 根据阈值判定
        threshold = thresholds[label]
        decisions[label] = fused_prob >= threshold

    return probs, fused_probs, decisions, matched_rules


def log_prediction(text, probs, decisions, matched_rules):
    """记录预测结果到日志"""
    os.makedirs('logs', exist_ok=True)
    log_path = 'logs/pred_log.csv'

    # 准备日志记录
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'text': text[:100],  # 限制文本长度
        'matched_rules': '; '.join(matched_rules) if matched_rules else 'None',
    }

    # 添加概率和决策
    for label in LABEL_NAMES:
        log_entry[f'prob_{label}'] = f"{probs[label]:.4f}"
        log_entry[f'decision_{label}'] = 'YES' if decisions[label] else 'NO'

    # 写入CSV
    file_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)


def main():
    """主应用"""
    st.set_page_config(
        page_title="多标签智能审核系统",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 用户评论多标签智能审核系统")
    st.markdown("基于预训练模型的中文评论多标签分类系统")

    # 侧边栏：模型选择
    st.sidebar.header("🧠 模型选择")
    available = list_available_best_models("outputs")
    if not available:
        st.sidebar.error("未在 outputs/ 下找到可用的 best_model")
        st.sidebar.info("请确保存在 outputs/<run_name>/best_model/（例如 outputs/roberta/best_model）")
        st.stop()

    # 默认优先 roberta，其次 macbert，否则用第一个
    preferred = ["roberta/best_model", "macbert/best_model", "best_model"]
    default_key = next((k for k in preferred if k in available), list(available.keys())[0])

    selected_key = st.sidebar.selectbox(
        "选择加载的模型",
        options=list(available.keys()),
        index=list(available.keys()).index(default_key),
        help="自动扫描 outputs/*/best_model"
    )
    model_path = available[selected_key]

    # 加载模型
    with st.spinner(f"正在加载模型：{selected_key} ..."):
        tokenizer, model = load_model(model_path)

    if tokenizer is None or model is None:
        st.stop()

    st.success(f"模型加载成功：{selected_key}")

    # 侧边栏：阈值设置
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 阈值设置")

    # 全局阈值
    global_threshold = st.sidebar.slider(
        "全局阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="应用到所有标签的基础阈值"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("各标签阈值微调")

    # 每个标签的阈值
    thresholds = {}
    for label in LABEL_NAMES:
        label_cn = LABEL_NAMES_CN[label]
        thresholds[label] = st.sidebar.slider(
            label_cn,
            min_value=0.0,
            max_value=1.0,
            value=global_threshold,
            step=0.05,
            key=f"threshold_{label}"
        )

    # 主界面：选项卡
    tab1, tab2 = st.tabs(["📝 单条预测", "📊 批量预测"])

    # Tab 1: 单条预测
    with tab1:
        st.header("单条文本审核")

        input_text = st.text_area(
            "请输入待审核的文本：",
            height=100,
            placeholder="例如：这是一条测试评论..."
        )

        if st.button("🔍 开始审核", type="primary"):
            if not input_text.strip():
                st.warning("请输入文本！")
            else:
                with st.spinner("正在分析..."):
                    probs, fused_probs, decisions, matched_rules = predict_single(
                        input_text, tokenizer, model, thresholds
                    )
                    log_prediction(input_text, fused_probs, decisions, matched_rules)

                st.subheader("📋 审核结果")

                if matched_rules:
                    st.warning("⚠️ 规则匹配结果：")
                    for rule in matched_rules:
                        st.write(f"- {rule}")
                else:
                    st.info("✓ 未匹配到特定规则")

                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🎯 模型预测概率")
                    for label in LABEL_NAMES:
                        label_cn = LABEL_NAMES_CN[label]
                        prob = probs[label]
                        st.progress(prob, text=f"{label_cn}: {prob:.2%}")

                with col2:
                    st.subheader("✅ 融合后判定结果")
                    for label in LABEL_NAMES:
                        label_cn = LABEL_NAMES_CN[label]
                        fused_prob = fused_probs[label]
                        decision = decisions[label]

                        if decision:
                            st.error(f"🚫 {label_cn}: {fused_prob:.2%} (触发)")
                        else:
                            st.success(f"✓ {label_cn}: {fused_prob:.2%} (正常)")

                st.markdown("---")
                if any(decisions.values()):
                    st.error("⛔ **综合判定：该内容存在违规风险，建议审核或屏蔽**")
                else:
                    st.success("✅ **综合判定：该内容未发现明显违规**")

    # Tab 2: 批量预测
    with tab2:
        st.header("批量文本审核")

        st.markdown("""
        上传包含 `text` 列的 CSV 文件进行批量审核。
        系统将为每条文本生成预测结果，并提供下载。
        """)

        uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')

            if 'text' not in df.columns:
                st.error("CSV文件必须包含 'text' 列！")
            else:
                st.info(f"已加载 {len(df)} 条文本")
                st.dataframe(df.head())

                if st.button("🚀 开始批量审核", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results = []

                    for idx, row in df.iterrows():
                        text = str(row['text'])

                        probs, fused_probs, decisions, matched_rules = predict_single(
                            text, tokenizer, model, thresholds
                        )

                        result_row = {'text': text}

                        for label in LABEL_NAMES:
                            result_row[f'{label}_prob'] = fused_probs[label]
                            result_row[f'{label}_decision'] = 'YES' if decisions[label] else 'NO'

                        result_row['matched_rules'] = '; '.join(matched_rules) if matched_rules else 'None'
                        result_row['has_violation'] = any(decisions.values())

                        results.append(result_row)

                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"处理进度: {idx + 1}/{len(df)}")

                    result_df = pd.DataFrame(results)

                    st.success("✅ 批量审核完成！")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总文本数", len(result_df))
                    with col2:
                        violation_count = int(result_df['has_violation'].sum())
                        st.metric("违规文本数", violation_count)
                    with col3:
                        violation_rate = violation_count / len(result_df) * 100
                        st.metric("违规率", f"{violation_rate:.1f}%")

                    st.subheader("结果预览")
                    st.dataframe(result_df.head(20))

                    csv_output = result_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下载完整结果 (CSV)",
                        data=csv_output,
                        file_name=f"audit_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **使用说明：**
    1. 在侧边栏选择要加载的模型（outputs/*/best_model）
    2. 调整各标签阈值
    3. 在“单条预测”中输入文本进行审核
    4. 在“批量预测”中上传CSV文件
    5. 单条预测会记录到 logs/pred_log.csv
    """)


if __name__ == '__main__':
    main()
