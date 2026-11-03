"""Audit completed paired pilot results and produce a Chinese Markdown report."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/runs/inmin_race_common_20261102'
METHODS = ('inmin_actor_critic_adapted', 'reinforce', 'local_random')
rows=[]
for model in ('resnet18','mobilenet_v2'):
    for seed in (311,312,313):
        folder=OUT/f'{model}_{seed}'
        protocol=json.loads((folder/'protocol.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        assert protocol['model']==model and protocol['search_seed']==seed and protocol['budget_per_method']==32
        assert set(protocol['methods'])==set(METHODS)
        identity=summary['common_task_identity']
        for method in METHODS:
            data=json.loads((folder/f'{method}.json').read_text())
            result=data['result']; assessment=result['best_assessment']
            assert data['common_task_identity']==identity and result['requests']==32
            assert result['actual_evaluations'] + result['cache_hits'] == 32
            if method=='inmin_actor_critic_adapted':
                assert len(result['trace'])==32 and data['policy_updates']==sum(row.get('accepted',False) for row in result['trace'])
            rows.append(dict(model=model,seed=seed,method=method,feasible=assessment['feasible'],
                cost=assessment['weighted_cost'],metrics=assessment['metrics'],
                evals=result['actual_evaluations'],hits=result['cache_hits'],seconds=data['total_wall_time_s'],
                updates=data.get('policy_updates')))
lines=['# INMIN actor-critic 改編與 RACE-RL：共同平台探索性比較','',
'本報告只比較受限共同空間上的 INMIN 線性 actor-critic 核心改編、RACE 原 CNN trainer，以及 local random。並非原生 INMIN v0.3 與 RACE 完整系統的公平定論。',
'固定各模型 INT8、64×64、相同 workload／硬體／成本／需求，平衡偏好、seed 311–313，每方法 32 個評估請求（含無效、重複與初始化）。每次試驗重新訓練；RACE episode=16。指標是未校準分析估算。',
'兩種 RL 的動作、候選、觀察、reward、初始化與 cache 規則不完全一致；INMIN 改編不含原生 seed、瓶頸特徵、複合動作與原生評估器。RACE 每 episode 重置 cache，另兩者每試次共享；計費同為請求數，實際後端評估（含物理無效）與時間另列。',
'PPA 使用 RACE 共同平台：面積僅 compute die，與 INMIN 原生 Area 定義不同；各模型限制取 research_exploration_20261030 的既有模型別研究條件，並非實際廠商規格。成本按共同限制正規化，愈低愈好，僅在同模型同偏好同條件且達標時比較。','',
'| 模型 | seed | 方法 | 達標 | 加權成本 | 延遲 ms | 功率 W | 面積 mm² | 實際評估 | 快取 | 時間 s |',
'|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    m=r['metrics']; fmt=lambda x: '—' if x is None else f'{x:.5f}'
    lines.append(f"| {r['model']} | {r['seed']} | {r['method']} | {'是' if r['feasible'] else '否'} | {fmt(r['cost'])} | {fmt(m['latency_s']*1000 if m['latency_s'] else None)} | {fmt(m['average_power_w'])} | {fmt(m['area_mm2'])} | {r['evals']} | {r['hits']} | {r['seconds']:.2f} |")
lines+=['','## 配對結果','']
for model in ('resnet18','mobilenet_v2'):
    pairs=[]
    for seed in (311,312,313):
        a=next(r for r in rows if r['model']==model and r['seed']==seed and r['method']=='inmin_actor_critic_adapted')
        b=next(r for r in rows if r['model']==model and r['seed']==seed and r['method']=='reinforce')
        if a['feasible'] and b['feasible']: pairs.append((a['cost'],b['cost']))
    lines.append(f"- {model}：兩者皆達標 {len(pairs)}/3；RACE 成本較低 {sum(b<a for a,b in pairs)} 組，改編 actor-critic 較低 {sum(a<b for a,b in pairs)} 組，持平 {sum(a==b for a,b in pairs)} 組。")
lines+=['','三個 seed 和 32 次請求只是小型 pilot；不能推論統計顯著、收斂、原版方法優劣或晶片 PPA。完整來源 hash、逐次紀錄及實際要求存於各組 protocol.json 與結果 JSON。',
'重現：在專案根目錄對模型 resnet18/mobilenet_v2 和 seed 311/312/313 執行 `tools/inmin_comparison/trial.py`，使用 --input-size 64 --budget 32 --rl-episode-size 16、各模型 limits 及三個 methods；每組使用新的輸出資料夾。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
