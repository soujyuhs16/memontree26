#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
系统验证脚本 - 检查所有依赖和文件是否就绪
运行此脚本以确保系统配置正确，然后再进行训练
"""

import sys
import os
import importlib


def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("   需要: Python 3.8+")
        return False


def check_dependencies():
    """检查必需的Python包"""
    print("\n检查Python依赖...")
    
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'numpy',
        'pandas',
        'sklearn',
        'streamlit',
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} (未安装)")
            missing.append(package)
    
    if missing:
        print(f"\n请安装缺失的包:")
        print(f"  pip install -r requirements.txt")
        return False
    
    return True


def check_dataset_files():
    """检查原始数据集文件"""
    print("\n检查原始数据集...")
    
    required_files = [
        'Dataset/AbuseSet/AbuseSet.csv',
        'Dataset/SexHarmset/SexHarmSet.csv',
        'Dataset/BiasSet/Bias_region.csv',
        'Dataset/BiasSet/BiasSet_genden.csv',
        'Dataset/BiasSet/Bias_race.csv',
        'Dataset/BiasSet/Bias_occupation.csv',
    ]
    
    missing = []
    for filepath in required_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filepath} ({size:,} bytes)")
        else:
            print(f"❌ {filepath} (不存在)")
            missing.append(filepath)
    
    if missing:
        print(f"\n缺少 {len(missing)} 个数据集文件")
        print("请确保Dataset目录包含所有必需的CSV文件")
        return False
    
    return True


def check_scripts():
    """检查主要脚本文件"""
    print("\n检查脚本文件...")
    
    scripts = [
        'build_multilabel_dataset.py',
        'train_multilabel.py',
        'app.py',
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print(f"✓ {script}")
        else:
            print(f"❌ {script} (不存在)")
            all_exist = False
    
    return all_exist


def check_directories():
    """检查必需的目录"""
    print("\n检查目录结构...")
    
    required_dirs = ['data', 'outputs', 'logs']
    
    for dirname in required_dirs:
        if not os.path.exists(dirname):
            print(f"  创建目录: {dirname}/")
            os.makedirs(dirname, exist_ok=True)
        print(f"✓ {dirname}/")
    
    return True


def check_cuda():
    """检查GPU/CUDA可用性"""
    print("\n检查GPU支持...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA可用")
            print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠ CUDA不可用 (将使用CPU训练，速度较慢)")
            return True
    except ImportError:
        print("⚠ 无法检查CUDA (torch未安装)")
        return True


def main():
    """主验证函数"""
    print("=" * 60)
    print("多标签智能审核系统 - 环境验证")
    print("=" * 60)
    
    checks = [
        ("Python版本", check_python_version),
        ("Python依赖", check_dependencies),
        ("数据集文件", check_dataset_files),
        ("脚本文件", check_scripts),
        ("目录结构", check_directories),
        ("GPU支持", check_cuda),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ 检查失败: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有检查通过！系统已就绪。")
        print("\n下一步操作:")
        print("  1. 生成数据集: python build_multilabel_dataset.py")
        print("  2. 训练模型:   python train_multilabel.py")
        print("  3. 启动应用:   streamlit run app.py")
    else:
        print("❌ 部分检查失败，请修复后再继续。")
        print("\n常见问题:")
        print("  - 缺少依赖: pip install -r requirements.txt")
        print("  - 缺少数据集: 确保Dataset/目录包含所有CSV文件")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
