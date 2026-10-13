"""Dependency-connected coupled atomic transactions; no PPA proposal ranking."""
from dataclasses import dataclass
from itertools import combinations
from .mapping import Mapping
from .tiles import partition, requirements


@dataclass(frozen=True)
class Action:
    root: str
    scope: tuple[str, ...]
    kind: str
    target: str
    count: int = 1
    compute_target: str = ""


def dependency_edges(workload,mapping):
    owners = {}
    for t in mapping.tiles:
        owners.setdefault(workload.operator_map[t.operator].output,[]).append(t)
    edges = set()
    for t in mapping.tiles:
        for name,region in requirements(workload,t):
            for s in owners.get(name,()):
                if s.output.intersection(region) is not None:
                    edges.add((s.id,t.id))
    return tuple(sorted(edges))


def scopes(workload,mapping,max_scope=4):
    """Bounded connected expansion from each root, independent of all oracle metrics."""
    edges = dependency_edges(workload,mapping)
    neighbors = {t.id:set() for t in mapping.tiles}
    for a,b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    result = set()
    for root in sorted(neighbors):
        frontier = {(root,)}
        for _ in range(max_scope):
            next_frontier = set()
            for scope in sorted(frontier):
                result.add((root,scope))
                candidates = set().union(*(neighbors[t] for t in scope))-set(scope)
                for t in sorted(candidates):
                    next_frontier.add(tuple(sorted((*scope,t))))
            frontier = next_frontier
    return tuple(sorted(result))


def apply(workload,hardware,mapping,action):
    by_id = {t.id:t for t in mapping.tiles}
    if action.root not in action.scope or not set(action.scope) <= set(by_id):
        raise ValueError("invalid action scope")
    selected = set(action.scope)
    if len(selected) != len(action.scope):
        raise ValueError("duplicate tile in action scope")
    visited = {action.root}
    edges = dependency_edges(workload,mapping)
    while True:
        expanded = visited | {b for a,b in edges if a in visited and b in selected} | {a for a,b in edges if b in visited and a in selected}
        if expanded == visited:
            break
        visited = expanded
    if visited != selected:
        raise ValueError("scope is not dependency connected")
    return _edit_connected(workload,hardware,mapping,action)


def _edit_connected(workload,hardware,mapping,action):
    """Apply a structurally generated connected scope without re-enumerating all roots."""
    by_id = {t.id:t for t in mapping.tiles}
    placement,storage = mapping.placements.copy(),mapping.storages.copy()
    tiles = mapping.tiles
    if action.kind == "cut_migrate":
        if action.target not in hardware.compute_ids:
            raise ValueError("invalid compute target")
        for t in action.scope: placement[t] = action.target
    elif action.kind == "memory_affinity":
        if action.target not in hardware.memory_ids:
            raise ValueError("invalid storage target")
        if action.compute_target not in hardware.compute_ids:
            raise ValueError("memory affinity must jointly select a compute target")
        for name in action.scope:
            placement[name] = action.compute_target
            op = workload.operator_map[by_id[name].operator]
            for tensor in (*op.inputs,op.output): storage[tensor] = action.target
            for parameter in workload.parameter_shapes(op): storage[parameter] = action.target
    elif action.kind == "work_share":
        # Rebuild whole selected operators; partial layer retile would lose ownership.
        operators = {by_id[t].operator for t in action.scope}
        if set(action.scope) != {t.id for t in mapping.tiles if t.operator in operators}:
            raise ValueError("work_share requires full operator ownership in scope")
        if action.target not in hardware.compute_ids:
            raise ValueError("invalid work_share target")
        counts = {o.id:sum(t.operator == o.id for t in mapping.tiles) for o in workload.operators}
        for o in operators: counts[o] = action.count
        rebuilt = partition(workload,counts)
        tiles = tuple(t for t in mapping.tiles if t.operator not in operators)+tuple(t for t in rebuilt if t.operator in operators)
        placement = {t.id:placement[t.id] for t in mapping.tiles if t.operator not in operators}
        start = hardware.compute_ids.index(action.target)
        for i,t in enumerate(t for t in tiles if t.operator in operators):
            placement[t.id] = hardware.compute_ids[(start+i)%len(hardware.compute_ids)]
    else:
        raise ValueError("unknown transaction kind")
    candidate = Mapping(tuple(sorted(tiles,key=lambda t:t.id)),tuple(sorted(placement.items())),tuple(sorted(storage.items())))
    candidate.validate(workload,hardware)
    return candidate


def legal_actions(workload,hardware,mapping,max_scope=4):
    """Structural legality only. All routes/traffic/lifetimes are recompiled by oracle."""
    result = []
    seen = set()
    for root,scope in scopes(workload,mapping,max_scope):
        candidates = [Action(root,scope,"cut_migrate",dst) for dst in hardware.compute_ids]
        candidates += [Action(root,scope,"memory_affinity",dst,compute_target=compute)
                       for dst in hardware.memory_ids for compute in hardware.compute_ids]
        candidates += [Action(root,scope,"work_share",dst,count) for dst in hardware.compute_ids for count in (1,2)]
        for a in candidates:
            try:
                # scopes() already establishes connectivity. Keep all mutation
                # and global coverage/placement/parameter validation unchanged.
                candidate = _edit_connected(workload,hardware,mapping,a)
            except ValueError:
                continue
            if candidate == mapping:
                continue
            key = (a.root,a.scope,a.kind,a.target,a.count,a.compute_target)
            if key not in seen:
                seen.add(key)
                result.append(a)
    return tuple(result)
