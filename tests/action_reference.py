"""Independent full compiler reference for testing structural action masks."""
from race_rl.actions import Action,scopes,apply


def exhaustive_actions(workload,hardware,mapping,max_scope):
    actions=[]
    for root,scope in scopes(workload,mapping,max_scope):
        candidates=[Action(root,scope,'cut_migrate',dst) for dst in hardware.compute_ids]
        candidates += [Action(root,scope,'memory_affinity',dst,compute_target=compute)
            for dst in hardware.memory_ids for compute in hardware.compute_ids]
        candidates += [Action(root,scope,'work_share',dst,count)
            for dst in hardware.compute_ids for count in (1,2)]
        for action in candidates:
            try:candidate=apply(workload,hardware,mapping,action)
            except ValueError:continue
            if candidate!=mapping:actions.append(action)
    return tuple(actions)
