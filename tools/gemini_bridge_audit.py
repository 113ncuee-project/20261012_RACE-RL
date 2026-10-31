#!/usr/bin/env python3
"""Check whether an exported Gemini mapping can be compared with a CNN workload.

This tool fails closed. It does not invent an equivalent chiplet mapping or PPA.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def validate_tiles(layer, batch):
    tiles = layer.get('output_tiles', [])
    shape = [batch, *layer['shape_chw']]
    if not tiles:
        return ['no exported output tiles']
    bounds = []
    volume = 0
    for tile in tiles:
        intervals = tile.get('bounds', [])
        if len(intervals) != 4 or any(len(pair) != 2 or not all(type(v) is int for v in pair)
                                      or not 0 <= pair[0] < pair[1] <= size
                                      for pair, size in zip(intervals, shape)):
            return ['output tile bounds invalid']
        volume += math.prod(end-start for start, end in intervals)
        for previous in bounds:
            if all(a[0] < b[1] and b[0] < a[1] for a,b in zip(intervals, previous)):
                return ['output tile bounds overlap']
        bounds.append(intervals)
    return [] if volume == math.prod(shape) else ['output tile coverage incomplete']


def audit(mapping, workload):
    if mapping.get('schema') != 1 or mapping.get('origin') != 'gemini_hpca2024':
        raise ValueError('unsupported or missing native Gemini mapping schema')
    layers = mapping['layers']
    operators = workload['operators']
    tensors = {item['id']:item for item in workload['tensors']}
    gemini_kinds = Counter(layer.get('kind', 'Unknown') for layer in layers)
    race_kinds = Counter(op['kind'] for op in operators)
    output_shapes = Counter(tuple(layer['shape_chw']) for layer in layers)
    race_shapes = Counter(tuple(tensors[op['output']]['shape'][1:]) for op in operators
                          if len(tensors[op['output']]['shape']) == 4)
    reasons = []
    if len(layers) != len(operators):
        reasons.append(f"operator count differs: Gemini {len(layers)}, RACE {len(operators)}")
    if output_shapes != race_shapes:
        reasons.append('operator output shape multisets differ')
    if race_kinds.get('ReLU', 0) and gemini_kinds.get('Other', 0) == 0:
        reasons.append('RACE contains explicit ReLU operations without a matching exported Gemini layer')
    if race_kinds.get('Linear', 0) and gemini_kinds.get('Linear', 0) == 0:
        reasons.append('RACE contains Linear operations absent from the exported Gemini graph')
    if race_kinds.get('BatchNorm', 0) and gemini_kinds.get('Other', 0) == 0:
        reasons.append('RACE contains BatchNorm operations without a matching exported Gemini layer')
    cores = mapping['mesh_x'] * mapping['mesh_y']
    chiplets = mapping['chiplet_x'] * mapping['chiplet_y']
    if cores != chiplets:
        reasons.append(f'Gemini has {cores} cores across {chiplets} chiplets; RACE placement indexes chiplets')
    multi_layer = sum(len(segment['partitions']) > 1 for segment in mapping['segments'])
    if multi_layer:
        reasons.append(f'{multi_layer} Gemini segments contain multiple layers; RACE models layer barriers')
    if any(segment.get('dram') for segment in mapping['segments']):
        reasons.append('Gemini DRAM selection cannot be represented by the single RACE DRAM scenario')
    tile_count = sum(len(layer.get('output_tiles', [])) for layer in layers)
    if not tile_count:
        reasons.append('exact output tile ownership is missing from the export')
    invalid_tiles = {layer['name']:issues for layer in layers
                     if (issues := validate_tiles(layer, mapping['batch']))}
    if invalid_tiles:
        reasons.append(f'Gemini output tile coverage failed for {len(invalid_tiles)} layers')
    reasons.extend(['Gemini per-core hardware is not established as equivalent to the RACE chiplet',
        'Gemini intra-core loop order and K blocking are not included in the mapping export',
        'Gemini internal search evaluations are not counted by the RACE environment',
        'Gemini and RACE cost, timing, and scheduling semantics have not been reconciled'])
    return {'status':'NOT_COMPARABLE', 'comparison_ready':False, 'reasons':reasons,
        'gemini':{'layers':len(layers), 'kinds':dict(gemini_kinds), 'cores':cores, 'chiplets':chiplets,
                  'multi_layer_segments':multi_layer, 'output_tiles':tile_count,
                  'tile_validation_errors':invalid_tiles,
                  'input_network_id':mapping['network_id']},
        'race':{'name':workload.get('name'), 'operators':len(operators), 'kinds':dict(race_kinds)},
        'scope':'audit only; no converted design or PPA result is emitted'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.mapping.read_text()), json.loads(args.workload.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists(): parser.error('choose a fresh output path')
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(f"{result['status']}: {args.out}; {len(result['reasons'])} incompatibilities")


if __name__ == '__main__':
    main()
