"""CPU PPO with full episodic gamma=1 returns and deterministic continuation."""
import random
import time
import numpy as np
import torch
from .policy import AutoregressiveActorCritic


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class RACETrainer:
    def __init__(self,*,seed=7,width=32,state_variant="full",lr=3e-4,clip=0.2,
                 epochs=4,entropy_coefficient=0.01,precision="float32"):
        seed_all(seed)
        if lr <= 0 or not 0 < clip < 1 or epochs < 1:
            raise ValueError("invalid PPO hyperparameters")
        if precision not in {"float32","float64"}:
            raise ValueError("unsupported policy precision")
        self.config = dict(seed=seed,width=width,state_variant=state_variant,lr=lr,clip=clip,
                           epochs=epochs,entropy_coefficient=entropy_coefficient,gamma=1.0,precision=precision)
        self.policy = AutoregressiveActorCritic(width,state_variant).to(
            dtype=torch.float64 if precision == "float64" else torch.float32)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=lr)
        self.episodes,self.training_evaluations,self.history = 0,0,[]

    def run_episode(self,env,*,learn=True):
        observation = env.reset()
        samples = []
        timings = {'action_selection_time_s':0.0,'environment_step_time_s':0.0,'ppo_update_time_s':0.0}
        def progress(stage):
            if not hasattr(env,'progress_path'):return
            from .serialization import save_json
            save_json(env.progress_path,{'stage':stage,'episode':self.episodes+1,
                'evaluations':env.oracle.evaluations,'evaluation_budget':env.oracle.budget,
                'requests':env.oracle.requests,'cache_hits':env.oracle.cache_hits,
                'current_ppa':{'power_w':env.current_record.average_power_w,'latency_s':env.current_record.latency_s,
                    'area_mm2':env.current_record.area_mm2},
                'last_reward':samples[-1][4] if samples else None,'best_objective':env.best_objective})
        while not env.done:
            progress('SELECTING_ACTION')
            started=time.perf_counter()
            action,log_probability,value = self.policy.select(observation)
            timings["action_selection_time_s"]+=time.perf_counter()-started
            index = observation.legal_actions.index(action)
            progress("ENVIRONMENT_STEP")
            started=time.perf_counter()
            next_o,r,done = env.step(action)
            timings["environment_step_time_s"]+=time.perf_counter()-started
            samples.append((observation,index,log_probability,value,r))
            observation = next_o
            progress("RECEIVED_FEEDBACK")
        stats = {"episode":self.episodes,"steps":len(samples),"oracle_evaluations":env.oracle.evaluations,
                 "oracle_requests":env.oracle.requests,"cache_hits":env.oracle.cache_hits,
                 "best_objective":env.best_objective,"return":sum(s[4] for s in samples),
                 "termination":env.termination,"parameter_updates":0}
        supports = [len(s[0].legal_actions) for s in samples]
        stats.update(initial_action_support=supports[0] if supports else len(observation.legal_actions),
            mean_action_support=float(np.mean(supports)) if supports else 0.0,
            max_action_support=max(supports,default=0))
        if learn and samples:
            progress("UPDATING_POLICY")
            started=time.perf_counter()
            self.update(samples,stats)
            timings["ppo_update_time_s"]=time.perf_counter()-started
        if hasattr(env,"progress_path"):
            stats.update(timings)
            progress("EPISODE_COMPLETED")
        if learn:
            self.episodes += 1
            self.training_evaluations += env.oracle.evaluations
            self.history.append(stats)
        return stats

    def update(self,samples,stats):
        returns,total = [],0.0
        for _,_,_,_,r in reversed(samples):
            total = r+total  # full Monte Carlo episodic return, gamma=1
            returns.append(total)
        dtype = next(self.policy.parameters()).dtype
        returns = torch.tensor(list(reversed(returns)),dtype=dtype)
        old_log = torch.tensor([s[2] for s in samples],dtype=dtype)
        old_values = torch.tensor([s[3] for s in samples],dtype=dtype)
        advantages = returns-old_values
        if len(samples) > 1:
            advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        for _ in range(self.config["epochs"]):
            logs,values,entropies = [],[],[]
            for o,index,*_ in samples:
                distribution,value = self.policy.distribution(o)
                logs.append(distribution.log_prob(torch.tensor(index)))
                values.append(value)
                entropies.append(distribution.entropy())
            ratio = torch.exp(torch.stack(logs)-old_log)
            clipped = torch.clamp(ratio,1-self.config["clip"],1+self.config["clip"])
            actor_loss = -torch.minimum(ratio*advantages,clipped*advantages).mean()
            value_loss = (torch.stack(values)-returns).square().mean()
            entropy = torch.stack(entropies).mean()
            loss = actor_loss+0.5*value_loss-self.config["entropy_coefficient"]*entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),0.5)
            self.optimizer.step()
            stats["parameter_updates"] += 1
        stats.update(actor_loss=actor_loss.item(),value_loss=value_loss.item(),entropy=entropy.item())

    def save(self,path,*,experiment_identity):
        torch.save({"schema":1,"config":self.config,"policy":self.policy.state_dict(),
            "optimizer":self.optimizer.state_dict(),"episodes":self.episodes,"history":self.history,
            "training_evaluations":self.training_evaluations,"torch_rng":torch.get_rng_state(),
            "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
            "experiment_identity":experiment_identity},path)

    @classmethod
    def load(cls,path,*,experiment_identity):
        # Only checkpoints generated locally by this project; pickle is never a remote input.
        saved = torch.load(path,map_location="cpu",weights_only=False)
        if saved["schema"] != 1 or saved["experiment_identity"] != experiment_identity:
            raise ValueError("checkpoint experiment identity mismatch")
        config = dict(saved["config"])
        config.pop("gamma")
        config.setdefault("precision","float32")  # historical checkpoints
        trainer = cls(**config)
        trainer.policy.load_state_dict(saved["policy"])
        trainer.optimizer.load_state_dict(saved["optimizer"])
        trainer.episodes,trainer.history = saved["episodes"],saved["history"]
        trainer.training_evaluations = saved["training_evaluations"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        return trainer
