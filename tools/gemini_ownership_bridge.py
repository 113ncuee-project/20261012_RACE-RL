#!/usr/bin/env python3
"""Find RACE layer plans with exactly the exported Gemini output ownership.

This checks only tile rectangles and their owners. It does not infer K blocking,
memory traffic, operator equivalence, schedule, hardware or overall PPA.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from race_rl.accelerators.cnn_design import LayerPlan, PROFILES, output_tiles
from race_rl.workload import Operator, Tensor, Workload


def matching_plans(workload, operator_id, gemini_layer, core_to_chiplet, chiplet_count):
    if not 1 <= chiplet_count <= 16:
        raise ValueError('RACE chiplet count must be 1–16')
    op = workload.operator_map[operator_id]
    shape = workload.tensor_map[op.output].shape
    if len(shape) != 4 or tuple(shape[1:]) != tuple(gemini_layer['shape_chw']):
        raise ValueError('operator output shapes differ')
    observed = Counter()
    if not gemini_layer.get('output_tiles'):
        raise ValueError('Gemini output tile ownership is missing')
    for tile in gemini_layer['output_tiles']:
        coordinate = ','.join(str(int(x)) for x in tile['core_xy'])
        if coordinate not in core_to_chiplet:
            raise ValueError(f'no explicit chiplet translation for Gemini core {coordinate}')
        owner = core_to_chiplet[coordinate]
        if type(owner) is not int or not 0 <= owner < chiplet_count:
            raise ValueError('core-to-chiplet translation outside chiplet count')
        observed[(tuple(tuple(pair) for pair in tile['bounds']), owner)] += 1
    candidates = []
    for h,w,c,k in PROFILES:
        for target in range(chiplet_count):
            for spread in (False, True):
                plan = LayerPlan(h,w,c,k,target,spread)
                proposed = Counter((tile.output.bounds,owner) for tile,owner in
                                   output_tiles(workload,op,plan,chiplet_count))
                if proposed == observed:
                    candidates.append({'h':h,'w':w,'c':c,'k':k,'target':target,'spread':spread})
    return candidates


def load_workload(path):
    raw = json.loads(path.read_text())
    return Workload(tuple(Tensor(**item) for item in raw['tensors']),
                    tuple(Operator(**item) for item in raw['operators']),
                    tuple(raw['outputs']), raw.get('name', 'custom'), raw.get('source_metadata', ()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--layer', required=True, help='Gemini layer name')
    parser.add_argument('--operator', required=True, help='explicit RACE operator id')
    parser.add_argument('--core-map', type=Path, required=True, help='JSON object: "x,y" to RACE chiplet id')
    parser.add_argument('--chiplets', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text())
    if mapping.get('origin') != 'gemini_hpca2024' or mapping.get('schema') != 1:
        raise ValueError('invalid Gemini mapping export')
    layers = [layer for layer in mapping['layers'] if layer['name'] == args.layer]
    if len(layers) != 1: raise ValueError('Gemini layer name must identify exactly one layer')
    workload = load_workload(args.workload)
    candidates = matching_plans(workload,args.operator,layers[0],json.loads(args.core_map.read_text()),args.chiplets)
    result = {'status':'OWNERSHIP_MATCH_ONLY' if candidates else 'NO_EXACT_OWNERSHIP_MATCH',
        'gemini_layer':args.layer, 'race_operator':args.operator, 'matching_plans':candidates,
        'limitations':['Only output tile rectangles and supplied core ownership are compared',
                       'K blocking, other operations, hardware, memory and schedule are unverified',
                       'No PPA equivalence or complete design is inferred']}
    if args.out.exists(): parser.error('choose a fresh output path')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(f"{result['status']}: {len(candidates)} candidate plans")


if __name__ == '__main__':
    main()
