#!/usr/bin/env python3
"""Compare initial count-action distributions with preference as the only change."""
import argparse,hashlib,json,math
from pathlib import Path
import torch
from race_rl.workload import Workload,Tensor,Operator
from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment
from race_rl.accelerators.cnn_ppa import load_cnn_cost_profile
from race_rl.accelerators.gemmini_model import PlatformScenario
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.requirements import PPARequirements
from race_rl.serialization import code_hashes
from race_rl.training import RACETrainer

PREFS=('balanced','power','performance','area')
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('experiments/runs/race_ppa_sensitivity_20261031/reference_v2'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 suite=read(a.source/'suite_protocol.json')
 if code_hashes()!=suite['source_hashes']:raise ValueError('current source differs from reference')
 results=[]
 for model in suite['models']:
  sample=a.source/'runs'/model/'balanced'/str(suite['seeds'][0]);raw=read(sample/'workload.json');proto=read(sample/'protocol.json')
  workload=Workload(tensors=tuple(Tensor(**x) for x in raw['tensors']),operators=tuple(Operator(**x) for x in raw['operators']),outputs=tuple(raw['outputs']),name=raw['name'],source_metadata=tuple(raw.get('source_metadata',())))
  memory=MemoryHierarchy.from_record(proto['memory']);platform=PlatformScenario.from_record(proto['platform']);profile=load_cnn_cost_profile(proto['cost_profile'],memory=memory)
  for seed in suite['seeds']:
   trainer=RACETrainer(seed=seed,precision='float64',policy_backend='gemmini_cnn');trainer.policy.eval();distributions={};features={}
   for pref in PREFS:
    req=PPARequirements(**dict(suite['limits'][model],preference=pref,weights=None)).require_vendor_inputs()
    env=GemminiCNNEnvironment(workload,requirements=req,memory=memory,platform=platform,cost_profile=profile,budget=1,max_scope=2);obs=env.reset()
    assert env.oracle.requests==0
    with torch.no_grad():
     _,_,g=trainer.policy.encoder(obs.graph);distribution=trainer.policy.categorical(trainer.policy.actor['count'](g));distributions[pref]=distribution.probs.detach().cpu().tolist();features[pref]=list(obs.graph.task_features)
   for pref in PREFS[1:]:
    x,y=distributions['balanced'],distributions[pref];tv=.5*sum(abs(a-b) for a,b in zip(x,y));results.append({'model':model,'seed':seed,'preference':pref,'total_variation':tv,'balanced_distribution':x,'preference_distribution':y,'balanced_features':features['balanced'],'preference_features':features[pref]})
  print(model,'count-action TV',[(r['preference'],round(r['total_variation'],6)) for r in results if r['model']==model and r['seed']==suite['seeds'][0]])
 output={'status':'INITIAL_POLICY_DIAGNOSTIC_NO_SEARCH_OR_TRAINING','notes':['exact same initial policy and empty environment per model and seed, only preference changes','compares 16-way initial compute-count marginal, not full action distribution','nonzero TV is conditioning, not proof that training or final designs respond'],'rows':results}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print('已儲存',a.out)
if __name__=='__main__':main()
