import networkx as nx
def map_blocks_to_graph(_blocks:list):
    """Create interim graph structure for beam search over blocks

    Parameter:
        _blocks: list of tuples, sorted by candidates
            Blocks are tuples with (length, (start positions for all witnesses))
            When aligning against an alignment tree, the sorted blocks need
                to be passed in because we can't sort by witness order

    Return:
        Networkx directed graph, possibly with cycles
        Nodes are blocks, node ids are the blocks themselves (tuples, except START and END)
        Edges are sequences, weighted by number of witnesses
            NB: We don't record which witnesses are on each edge (we could label the edges if we care)
        Serves as input into beam search to find best route between START and END
        Always connected because all paths pass through at least one block
    """
    result = nx.DiGraph()
    result.add_node(0, label="START")
    result.add_node(1, label="END")
    for _witness_data in _blocks:
        for _block in _witness_data: # automatically removes duplicates
            result.add_node(_block,  label=_block)
    return result

