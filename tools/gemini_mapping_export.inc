// Read-only export of Gemini's final mapping; included inside main().
if (argc > 2) {
    std::ofstream bridge(argv[2]);
    if (!bridge) throw std::runtime_error("cannot open mapping export");
    bridge << "{\"schema\":1,\"origin\":\"gemini_hpca2024\",\"network_id\":" << nn
           << ",\"mesh_x\":" << xx << ",\"mesh_y\":" << yy
           << ",\"chiplet_x\":" << xcut << ",\"chiplet_y\":" << ycut
           << ",\"mac_dim\":" << _mac_dim << ",\"ul3_kib\":" << _ul3
           << ",\"noc_bw\":" << _NoC_bw << ",\"nop_bw\":" << _serdes_lane
           << ",\"batch\":" << bb << ",\"layers\":[";
    for (unsigned int i=0; i<network->len(); ++i) {
        if(i) bridge << ',';
        const auto &node = network->getNode(i);
        const auto &shape = node.layer().ofmap_shape();
        const char *kind = dynamic_cast<const FCLayer*>(&node.layer()) ? "Linear" :
            dynamic_cast<const GroupConvLayer*>(&node.layer()) ? "GroupConv" :
            dynamic_cast<const ConvLayer*>(&node.layer()) ? "Conv" :
            dynamic_cast<const PoolingLayer*>(&node.layer()) ? "Pool" :
            dynamic_cast<const EltwiseLayer*>(&node.layer()) ? "Eltwise" : "Other";
        bridge << "{\"id\":" << i << ",\"name\":\"" << node.name()
               << "\",\"kind\":\"" << kind
               << "\",\"shape_chw\":[" << shape.c << ',' << shape.h << ',' << shape.w
               << "],\"predecessors\":[";
        bool first = true;
        FOR_BITSET(pred, node.getPrevs()) {
            if(!first) bridge << ',';
            first = false;
            bridge << pred;
        }
        bridge << "],\"output_tiles\":[";
        const LNode* lnode = scheme->schtree_spm->get_lnode_by_id(i);
        bool first_tile = true;
        if (lnode) {
            const auto &tiles = lnode->get_place_sch().getOfmL();
            for (auto tile : tiles) {
                const auto &r = tile.first;
                if (r.is_empty()) continue;
                if (!first_tile) bridge << ',';
                first_tile = false;
                bridge << "{\"bounds\":[[" << r.b.from << ',' << r.b.to << "],["
                    << r.c.from << ',' << r.c.to << "],[" << r.h.from << ',' << r.h.to
                    << "],[" << r.w.from << ',' << r.w.to << "]],\"core_xy\":["
                    << tile.second.x << ',' << tile.second.y << "]}";
            }
        }
        bridge << "]}";
    }
    bridge << "],\"segments\":[";
    for (size_t s=0; s<scheme->placement.size(); ++s) {
        if(s) bridge << ',';
        const auto &place=scheme->placement[s];
        const auto &parts=place.get_partition();
        const auto &cores=place.get_placement();
        bridge << "{\"partitions\":[";
        for (size_t p=0; p<parts.size(); ++p) {
            if(p) bridge << ',';
            const auto &part=parts[p];
            bridge << "{\"layer_id\":" << part.layerno << ",\"b\":" << part.b
                   << ",\"c\":" << part.c << ",\"h\":" << part.h
                   << ",\"w\":" << part.w << ",\"size\":" << part.siz << '}';
        }
        bridge << "],\"cores\":[";
        for(size_t i=0; i<cores.size(); ++i) {
            if(i) bridge << ',';
            bridge << "{\"core_index\":" << i << ",\"layer_id\":" << cores[i].layerno
                   << ",\"partition_index\":" << cores[i].partno << '}';
        }
        bridge << "],\"dram\":[";
        bool first_dram = true;
        for (const auto &item : place.layer_DRAM) {
            if(!first_dram) bridge << ',';
            first_dram = false;
            bridge << "{\"layer_id\":" << item.first << ",\"ports\":[";
            for (size_t d=0; d<item.second.size(); ++d) {
                if(d) bridge << ',';
                bridge << static_cast<int>(item.second[d]);
            }
            bridge << "]}";
        }
        bridge << "]}";
    }
    bridge << "]}\n";
    if (!bridge) throw std::runtime_error("cannot write mapping export");
}
