"""Aggregate actual selected-design mapping and traffic into a browser-sized view."""
from collections import defaultdict


def build_design_view(hardware,mapping,evaluation,compliant):
    placed=dict(mapping['placement']);tile_map={t['id']:t for t in mapping['tiles']}
    assignments=defaultdict(list)
    for tid,node in mapping['placement']:
        assignments[node].append({'tile':tid,'operator':tile_map[tid]['operator'],'bounds':tile_map[tid]['output']['bounds']})
    nodes=[]
    for n in hardware['nodes']:
        tiles=assignments[n['id']];ops=defaultdict(int)
        for tile in tiles:ops[tile['operator']]+=1
        nodes.append({'id':n['id'],'x_mm':n['x_mm'],'y_mm':n['y_mm'],'width_mm':n['width_mm'],
            'height_mm':n['height_mm'],'tile_count':len(tiles),'operators':dict(ops),'tiles':tiles})
    links={l['id']:{'id':l['id'],'a':l['a'],'b':l['b'],'bandwidth_Bps':l['bandwidth_Bps'],
        'a_to_b_bytes':0,'b_to_a_bytes':0,'flows':0} for l in hardware['links']}
    ports={p['id']:{**p,'read_bytes':0,'write_bytes':0} for p in hardware.get('memory_ports',[])}
    flows=[]
    for f in evaluation['traffic_ledger']:
        size=f['bytes']
        for lid,a,b in f['route']:
            link=links[lid]
            key='a_to_b_bytes' if (a,b)==(link['a'],link['b']) else 'b_to_a_bytes'
            link[key]+=size;link['flows']+=1
        if f['source'] in ports:ports[f['source']]['read_bytes']+=size
        if f['destination'] in ports:ports[f['destination']]['write_bytes']+=size
        flows.append({'source':f['source'],'destination':f['destination'],'tensor':f['tensor'],'bytes':size,
            'route':[{'link':lid,'from':a,'to':b} for lid,a,b in f['route']]})
    groups=defaultdict(lambda:defaultdict(int))
    for tile,node in placed.items():groups[tile_map[tile]['operator']][node]+=1
    return {'compliant':compliant,'compute_count':len(nodes),'nodes':nodes,'links':list(links.values()),
        'memory_ports':list(ports.values()),'storage':[{'object':name,'endpoint':node} for name,node in mapping['storage']],
        'operator_placement':[{'operator':op,'tiles_by_node':dict(by)} for op,by in groups.items()],
        'largest_transfers':sorted(flows,key=lambda f:f['bytes'],reverse=True)[:30],
        'transfer_count':len(flows),'total_mesh_hop_bytes':sum(l['a_to_b_bytes']+l['b_to_a_bytes'] for l in links.values()),
        'traffic_definition':'cumulative bytes per directed physical hop during one inference; not simultaneous bandwidth or point-to-point payload sum'}
