"""
简化版超参数搜索
逐步调参，每次只调整1-2个参数，避免过拟合
"""

import subprocess
import re
import time
import json

def run_single_experiment(exp_name, param_changes):
    """
    运行单个实验
    
    Args:
        exp_name: 实验名称
        param_changes: 要修改的参数字典，如 {'hidden_size': 128}
    """
    print(f"\n{'='*80}")
    print(f"实验: {exp_name}")
    print(f"参数变更: {param_changes}")
    print("="*80)
    
    config_path = "config/model_gru.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for key, value in param_changes.items():
        pattern = rf'^(\s+{key}:\s*).*'
        replacement = f'  {key}: {value}'
        content = re.sub(pattern, replacement, content)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    cmd = "conda activate qlib_zhengshi; cd d:\\lianghuatouzi\\Qlib1124\\project; python run_train.py --config config/pipeline.yaml"
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    output = result.stdout + result.stderr
    
    icir_match = re.search(r'GRU IC 汇总.*?ICIR=([0-9.]+)', output)
    ic_mean_match = re.search(r'GRU IC 汇总.*?mean=([0-9.]+)', output)
    ic_std_match = re.search(r'GRU IC 汇总.*?std=([0-9.]+)', output)
    
    if icir_match:
        icir = float(icir_match.group(1))
        ic_mean = float(ic_mean_match.group(1)) if ic_mean_match else 0
        ic_std = float(ic_std_match.group(1)) if ic_std_match else 0
        print(f"\n结果: IC={ic_mean:.6f}, IC_std={ic_std:.6f}, ICIR={icir:.6f}")
        return {"icir": icir, "ic_mean": ic_mean, "ic_std": ic_std}
    else:
        print("未找到结果")
        return None

if __name__ == "__main__":
    print("="*80)
    print("逐步调参 - 避免过拟合")
    print("="*80)
    
    baseline_icir = 0.755336
    
    experiments = [
        {
            "name": "增加模型容量",
            "desc": "hidden_size: 64->128, 增加模型表达能力",
            "params": {"hidden_size": 128}
        },
        {
            "name": "增加网络深度",
            "desc": "num_layers: 2->3, 增加非线性能力",
            "params": {"num_layers": 3}
        },
        {
            "name": "增强正则化",
            "desc": "dropout: 0.3->0.5, 防止过拟合",
            "params": {"dropout": 0.5}
        },
        {
            "name": "提高学习率",
            "desc": "lr: 3e-4->1e-3, 加快收敛",
            "params": {"lr": 0.001}
        },
        {
            "name": "增强非对称损失",
            "desc": "gamma: 3.0->5.0, 更重视正向收益",
            "params": {"gamma": 5.0}
        },
        {
            "name": "增加序列长度",
            "desc": "seq_len: 5->10, 捕获更长期依赖",
            "params": {"seq_len": 10}
        },
    ]
    
    results = []
    
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {exp['name']}: {exp['desc']}")
        
        result = run_single_experiment(exp['name'], exp['params'])
        
        if result:
            result['name'] = exp['name']
            result['desc'] = exp['desc']
            result['params'] = exp['params']
            results.append(result)
        
        if i < len(experiments) - 1:
            print("\n等待3秒...")
            time.sleep(3)
    
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    
    for r in sorted(results, key=lambda x: x['icir'], reverse=True):
        improvement = ((r['icir'] - baseline_icir) / baseline_icir * 100)
        print(f"{r['name']}: ICIR={r['icir']:.6f} (提升{improvement:.1f}%), IC={r['ic_mean']:.6f}±{r['ic_std']:.6f}")
    
    if results:
        best = max(results, key=lambda x: x['icir'])
        print(f"\n最佳参数组合:")
        print(f"  参数: {best['params']}")
        print(f"  ICIR={best['icir']:.6f} (相比基线提升{((best['icir'] - baseline_icir) / baseline_icir * 100):.1f}%)")
        
        with open('hyperparameter_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n结果已保存到 hyperparameter_results.json")
