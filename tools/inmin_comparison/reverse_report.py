"""Validate and summarize the reverse INMIN-backed pilot."""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'experiments/runs/inmin_race_reverse_20261102'
methods = ('bottleneck_rl','race_reinforce_adapted')
lines = ['# 反向適應：RACE REINFORCE 與 INMIN 共用 INMIN v0.3 平台', '',
'兩組使用相同 INMIN v0.3 toy workload、所有 1–16 Compute 矩形候選、原生評估器、共同 seed、瓶頸候選生成與 283 維特徵，以及 INMIN 原 reward。Batch 1，均衡偏好，無額外 PPA 硬上限。每組最多 32 個完整 PPA，含相同的 4 個 seed；時間上限也相同。',
'RACE 改編版保留實際 RACETrainer 的 REINFORCE reward-to-go、Adam 3e-4、float64、無 critic 與梯度裁切；使用新的線性候選 actor 與 INMIN 觀察、動作、reward、初始解及環境，並非完整 RACE。INMIN 為原生線性 actor-critic、逐步 TD 更新；RACE 改編每回合更新，末尾不完整回合也更新。無效評估不更新。各方法依原本執行方式作推理和更新，cache、實際完整 PPA、timeout、時間均列示。',
'模型成本是未校準分析估算。與正向比較採用不同硬體、Area 定義和評分基準，數字不可跨平台直接相減。', '',
'| seed | 方法 | 完整 PPA（含 seed） | cache hits | timeout | 更新次數 | 最佳成本 | 相對 seed 改善 | latency µs | power W | area mm² | 時間 s |',
'|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for seed in (311,312,313):
    folder=OUT/f'toy_{seed}'
    config=json.loads((folder/'effective_config.json').read_text())
    state=json.loads((folder/'run_state.json').read_text())
    provenance=json.loads((folder/'comparison_provenance.json').read_text())
    result=json.loads((folder/'summary.json').read_text())
    assert state['status']=='completed' and state['source_stable']
    assert config['seed']==seed and config['queries']==32 and config['model']=='toy'
    assert set(result)==set(methods)
    for method in methods:
        r=result[method]; best=r['best']; cp=json.loads((folder/method/'checkpoint.json').read_text())
        assert r['full_ppa_completed']<=32 and r['common_seed_full_ppa']==4
        assert r['search_full_ppa']+r['common_seed_full_ppa']==r['full_ppa_completed']
        assert cp['policy']['updates']==r['policy_updates']
        if method=='race_reinforce_adapted':
            assert cp['policy']['kind']=='race_reinforce_candidate_adapter'
            assert cp['policy']['pending_steps']==0
        lines.append(f"| {seed} | {method} | {r['full_ppa_completed']} | {r['cache_hits']} | {r['oracle_timeouts']} | {r['policy_updates']} | {r['best_cost']:.9f} | {r['improvement_over_best_seed_pct']:.3f}% | {best['latency_s']*1e6:.6f} | {best['power_w']:.6f} | {best['area_mm2']:.6f} | {r['wall_s_including_common_seed_charge']:.2f} |")
probe=OUT/'resnet18_probe'
if (probe/'summary.json').exists() and (probe/'run_state.json').exists():
    probe_state=json.loads((probe/'run_state.json').read_text())
    if probe_state.get('status')=='completed':
        probe_results=json.loads((probe/'summary.json').read_text())
        lines.extend(['', '## ResNet-18 64×64 接線短測', '',
                      '單 seed 311，最多 8 次完整 PPA 含 2 次共用 seed；僅供接線與預算檢查，不足以衡量學習效益。', '',
                      '| 方法 | 完整 PPA | policy 更新 | 最佳成本 | 延遲 ms | 功率 W | 面積 mm² | 狀態 |',
                      '|---|---:|---:|---:|---:|---:|---:|---|'])
        for method in methods:
            r=probe_results[method]; b=r['best']
            assert r['full_ppa_completed']<=8 and r['common_seed_full_ppa']==2
            if b:
                lines.append(f"| {method} | {r['full_ppa_completed']} | {r['policy_updates']} | {r['best_cost']:.9f} | {b['latency_s']*1000:.6f} | {b['power_w']:.6f} | {b['area_mm2']:.6f} | {r['status']} |")
lines += ['', '這是三 seed、小型 toy、單一均衡偏好的探索性測試。相同上限不保證相同軌跡或完成次數；平手不證明兩個方法等價，也不能代表真實 CNN 或原生 RACE 的表現。逐動作、seed 和 source hash 均在各試驗資料夾。',
'更新次數意義不同：INMIN 每個有效步驟更新，RACE 改編每回合更新；不能把數量直接當作學習品質。固定同一 PPA 模型也無法證明模型對真實晶片準確。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
