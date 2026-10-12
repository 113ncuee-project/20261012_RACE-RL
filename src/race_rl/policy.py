"""Learnable relational message passing and hierarchical masked action probabilities."""
import torch
from torch import nn
from .observation import TILE_FEATURE_SIZE


class RelationalEncoder(nn.Module):
    def __init__(self,width=32,variant="full"):
        super().__init__()
        if variant not in {"full","compact","dnngraph","hardwaregraph","no_attribution","no_criticality"}:
            raise ValueError("unknown state ablation")
        self.variant,self.width = variant,width
        self.tile = nn.Sequential(nn.Linear(TILE_FEATURE_SIZE,width),nn.Tanh())
        self.hardware = nn.Sequential(nn.Linear(9,width),nn.Tanh())
        self.dnn_message = nn.Linear(width,width,bias=False)
        self.hw_message = nn.Linear(width,width,bias=False)
        self.memory_message = nn.Linear(width,width,bias=False)
        self.cross_message = nn.Linear(width,width,bias=False)
        self.link_message = nn.Linear(4,width,bias=False)
        self.global_net = nn.Sequential(nn.Linear(8+2*width,width),nn.Tanh())

    def forward(self,o):
        if o.schema_version != 3:
            raise ValueError("incompatible observation schema; old checkpoints require their frozen source")
        dtype = self.tile[0].weight.dtype
        tf = torch.tensor(o.tile_features,dtype=dtype)
        hf = torch.tensor(o.hardware_features,dtype=dtype)
        ef = torch.tensor(o.hardware_edge_features,dtype=dtype).reshape(-1,4)
        if self.variant == "no_criticality":
            tf[:,-2:] = 0
            ef[:,-1] = 0
        t,h = self.tile(tf),self.hardware(hf)
        tm,hm = torch.zeros_like(t),torch.zeros_like(h)
        if self.variant in {"full","dnngraph","no_attribution","no_criticality"}:
            for a,b in o.dependency_edge_index:
                tm[b] = tm[b]+self.dnn_message(t[a])
                tm[a] = tm[a]+self.dnn_message(t[b])
        if self.variant in {"full","hardwaregraph","no_attribution","no_criticality"}:
            for a,b in o.hardware_edge_index:
                hm[b] = hm[b]+self.hw_message(h[a])
            for a,b in o.memory_port_edge_index:
                hm[b] = hm[b]+self.memory_message(h[a])
        if self.variant in {"full","no_attribution","no_criticality"}:
            for a,b in o.placement_edge_index:
                tm[a] = tm[a]+self.cross_message(h[b])
                hm[b] = hm[b]+self.cross_message(t[a])
        if self.variant in {"full","no_criticality"}:
            seen = set()
            for d,l,flow,_ in o.dependency_link_relations:
                # Shared flow-link attributed once per dependency, without duplicating energy ledger.
                if (d,l,flow) in seen: continue
                seen.add((d,l,flow))
                a,b = o.dependency_edge_index[d]
                message = self.link_message(ef[l])
                tm[a],tm[b] = tm[a]+message,tm[b]+message
        t,h = torch.tanh(t+tm),torch.tanh(h+hm)
        if self.variant == "compact":
            t = t.mean(0).expand_as(t)
            h = h.mean(0).expand_as(h)
        glob = torch.tensor(o.global_features,dtype=dtype)
        # Raw SI energy/latency are transformed without fitted/test-set statistics.
        glob[4:6] = torch.log(torch.clamp(glob[4:6],min=1e-30))/30
        g = self.global_net(torch.cat((glob,t.mean(0),h.mean(0))))
        return t,h,g


class AutoregressiveActorCritic(nn.Module):
    def __init__(self,width=32,state_variant="full"):
        super().__init__()
        self.encoder = RelationalEncoder(width,state_variant)
        self.root_score = nn.Linear(2*width,1)
        self.scope_score = nn.Linear(3*width,1)
        self.type_embeddings = nn.Embedding(3,width)
        self.type_score = nn.Linear(4*width,1)
        self.parameter_score = nn.Sequential(nn.Linear(6*width+1,width),nn.Tanh(),nn.Linear(width,1))
        self.critic = nn.Linear(width,1)
        self.design_family = nn.Linear(width,3)
        self.count_embedding = nn.Sequential(nn.Linear(4,width),nn.Tanh())
        self.count_score = nn.Linear(2*width,1)
        self.retile_score = nn.Sequential(nn.Linear(3*width+2,width),nn.Tanh(),nn.Linear(width,1))

    def distribution(self,o):
        from .architecture import DesignObservation
        if isinstance(o,DesignObservation):
            return self.design_distribution(o)
        if not o.legal_actions:
            raise ValueError("no legal action distribution")
        t,h,g = self.encoder(o)
        ti = {name:i for i,name in enumerate(o.tile_ids)}
        hi = {name:i for i,name in enumerate(o.hardware_ids)}
        actions = [a for a,mask in zip(o.legal_actions,o.legal_action_masks) if mask]
        if not actions:
            raise ValueError("all actions masked")
        roots = sorted({a.root for a in actions})
        root_logits = torch.stack([self.root_score(torch.cat((g,t[ti[r]]))).squeeze() for r in roots])
        root_logs = dict(zip(roots,torch.log_softmax(root_logits,0)))
        type_ids = {"cut_migrate":0,"work_share":1,"memory_affinity":2}
        leaf_logs = {}
        for root in roots:
            matching = [a for a in actions if a.root == root]
            scopes = sorted({a.scope for a in matching})
            scope_vectors = {s:t[[ti[name] for name in s]].mean(0) for s in scopes}
            scope_logits = torch.stack([self.scope_score(torch.cat((g,t[ti[root]],scope_vectors[s]))).squeeze() for s in scopes])
            for scope,scope_log in zip(scopes,torch.log_softmax(scope_logits,0)):
                sv = scope_vectors[scope]
                kinds = sorted({a.kind for a in matching if a.scope == scope})
                type_vectors = {k:self.type_embeddings.weight[type_ids[k]] for k in kinds}
                type_logits = torch.stack([self.type_score(torch.cat((g,t[ti[root]],sv,type_vectors[k]))).squeeze() for k in kinds])
                for kind,type_log in zip(kinds,torch.log_softmax(type_logits,0)):
                    params = [a for a in matching if a.scope == scope and a.kind == kind]
                    param_logits = torch.stack([self.parameter_score(torch.cat((g,t[ti[root]],sv,
                        type_vectors[kind],h[hi[a.target]],h[hi[a.compute_target]] if a.compute_target else torch.zeros_like(g),
                        g.new_tensor([a.count/4])))).squeeze() for a in params])
                    for a,param_log in zip(params,torch.log_softmax(param_logits,0)):
                        leaf_logs[a] = root_logs[root]+scope_log+type_log+param_log
        joint = torch.stack([leaf_logs[a] if mask else g.new_tensor(float('-inf'))
                             for a,mask in zip(o.legal_actions,o.legal_action_masks)])
        distribution = torch.distributions.Categorical(logits=joint)
        return distribution,self.critic(g).squeeze()

    def design_distribution(self,o):
        from .architecture import CountAction,RetileAction,IndependentAction
        t,h,g = self.encoder(o.graph)
        families = [[],[],[]]
        for i,a in enumerate(o.legal_actions):
            families[1 if isinstance(a,CountAction) else 2 if isinstance(a,(RetileAction,IndependentAction)) else 0].append(i)
        available = [k for k,indices in enumerate(families) if any(o.legal_action_masks[i] for i in indices)]
        if not available: raise ValueError('all design actions masked')
        # Exploration is part of the actual distribution used for PPO ratios.
        def mixed(logits):
            return 0.95*torch.softmax(logits,0)+0.05/len(logits)
        gate = mixed(self.design_family(g)[available])
        log_gate = dict(zip(available,torch.log(gate)))
        logs = [g.new_tensor(float('-inf')) for _ in o.legal_actions]
        ti = {name:i for i,name in enumerate(o.graph.tile_ids)}
        hi = {name:i for i,name in enumerate(o.graph.hardware_ids)}
        for family in available:
            indices = [i for i in families[family] if o.legal_action_masks[i]]
            if family == 0:
                from dataclasses import replace
                masks = tuple(o.legal_action_masks[i] for i in families[0])
                distribution,_ = self.distribution(replace(o.graph,legal_action_masks=masks))
                probs = distribution.probs[[families[0].index(i) for i in indices]]
                conditional = 0.95*probs+0.05/len(indices)
            elif family == 1:
                vectors = []
                for i in indices:
                    feature = o.count_features[families[1].index(i)]
                    embedded = self.count_embedding(g.new_tensor(feature))
                    vectors.append(self.count_score(torch.cat((g,embedded))).squeeze())
                conditional = mixed(torch.stack(vectors))
            else:
                vectors = []
                for i in indices:
                    ids,target,count,spread = o.retile_features[families[2].index(i)]
                    group = [ti[name] for name in ids]
                    vectors.append(self.retile_score(torch.cat((g,t[group].mean(0),h[hi[target]],
                        g.new_tensor([count,spread])))).squeeze())
                conditional = mixed(torch.stack(vectors))
            for i,p in zip(indices,conditional): logs[i] = log_gate[family]+torch.log(p)
        return torch.distributions.Categorical(logits=torch.stack(logs)),self.critic(g).squeeze()

    @torch.no_grad()
    def select(self,o):
        distribution,value = self.distribution(o)
        index = distribution.sample()
        return o.legal_actions[index.item()],distribution.log_prob(index).item(),value.item()
