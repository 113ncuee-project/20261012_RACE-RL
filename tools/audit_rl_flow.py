"""Verify one selected action, one environment feedback; prohibit actor PPA calls."""
import argparse
from pathlib import Path
from unittest.mock import patch
from race_rl.architecture import DesignEnvironment
from race_rl.examples import toy_workload
from race_rl.training import RACETrainer
from race_rl.oracle import Oracle
from race_rl.serialization import save_json,code_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    env=DesignEnvironment(toy_workload(),compute_counts=(2,1,*range(3,17)),budget=4)
    with patch.object(Oracle,'__init__',side_effect=AssertionError('native setup before action')):
        o=env.reset()
    assert env.oracle.evaluations==env.oracle.requests==0
    trainer=RACETrainer(seed=7,width=8,precision='float64');rows=[]
    while not env.done:
        before=(env.oracle.evaluations,env.oracle.requests)
        with patch.object(env.oracle,'evaluate',side_effect=AssertionError('actor called PPA')):
            with patch.object(Oracle,'evaluate',side_effect=AssertionError('actor called backend PPA')):
                action,log_pi,value=trainer.policy.select(o)
        assert before==(env.oracle.evaluations,env.oracle.requests)
        assert value is None and not hasattr(trainer.policy,'critic')
        evaluate=env.oracle.evaluate
        with patch.object(env.oracle,'evaluate',wraps=evaluate) as called:
            next_o,r,done=env.step(action)
        assert called.call_count==1
        rows.append({'action':action,'actor_oracle_calls':0,'environment_evaluation_entry_calls':called.call_count,
            'reward':r,'before_evaluations':before[0],'after_evaluations':env.oracle.evaluations,
            'before_requests':before[1],'after_requests':env.oracle.requests,
            'next_observation_compute_count':next_o.compute_count,'next_observation_current_ppa':{
                'average_power_w':next_o.graph.global_features[4],'latency_s':next_o.graph.global_features[5]},
            'candidate_ppa_from_environment':env.trace[-1]['ppa_assessment'],
            'next_observation_last_reward':next_o.graph.global_features[9],'done':done})
        o=next_o
    save_json(args.out,{'status':'PASSED','algorithm':'reinforce','critic_used':False,'source_hashes':code_hashes(),'initial_evaluations':0,'steps':rows,
        'contract':'one Agent action -> environment -> one reward and next observation; actor cannot query PPA'})
    print(f'RL flow audit passed: {len(rows)} steps; no actor PPA calls')


if __name__=='__main__':main()
