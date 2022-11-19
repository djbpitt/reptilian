"""Tree of nodes

TODO: Replace our own classes with networkx
Nodes have a type property: aligned, unaligned, branching
Each of the three types requires specific properties and prohibits others
TODO: Can we formalize those requirements and prohibitions? Can a networkx
    property be a complex object?
"""
import networkx as nx
from collections import deque
from typing import List
from create_blocks import *

def create_tree() -> nx.DiGraph:
    """Create new DiGraph with no nodes

    Graph will represent alignment tree"""
    _G = nx.DiGraph()
    return _G


def expand_node(_graph: nx.DiGraph, _node_ids: deque, _token_array, _token_membership_array, _witness_count):
    """Expand and then remove head of deque

    Create suffix array and LCP array

    Add new nodes to tail of deque
    No return because graph and deque are both modified in place
    """
    _sa = create_suffix_array(_token_array)
    _frequent_sequences = create_blocks(_sa,_token_membership_array, _witness_count)
    _largest_blocks = find_longest_sequences(_frequent_sequences, _sa)
    _block_offsets_by_witness, _witness_offsets_to_blocks, _first_token_offset_in_block_by_witness, \
        _first_absolute_token_by_witness, _score_by_block = \
        prepare_for_beam_search(_witness_count, _token_membership_array, _largest_blocks)
    _finished = perform_beam_search(_witness_count, _largest_blocks, _block_offsets_by_witness,
                                    _witness_offsets_to_blocks, _score_by_block)
    # debug output; remove before continuing
    # Resume with Jupyter cell to "Create tree to represent alignment
    for pos, f in enumerate(_finished):
        print(pos, sum([_largest_blocks[b][0] for b in f.path]), len(f.path))
        # print(f)