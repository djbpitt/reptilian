"""Tree of nodes

Node properties:
    All nodes:
        id:int : consecutive integers
        type:str : aligned, unaligned, branching, potential (= unexpended)
        token_ranges:list[Tuple] : offsets in global token array
        children:list[Int] : ids of child nodes
    TODO: children make sense only for branching nodes, but we don't know how to
        add an attribute for some nodes but not others, so we add an empty list
        for all. We are embarrassed.
"""
import networkx as nx
from collections import deque
from typing import List
from create_blocks import *
import graphviz

def create_tree() -> nx.DiGraph:
    """Create new DiGraph with no nodes

    Graph will represent alignment tree"""
    _G = nx.DiGraph()
    return _G


def expand_node(_graph: nx.DiGraph, _node_ids: deque, _token_array, _token_membership_array, _witness_count: int):
    """Expand and then remove head of deque

    Find best path through blocks:
        Create suffix array and LCP array
        Find longest full-depth, non-repeating frequent sequences
        Find largest blocks
        Use beam search to find best path through blocks:
            most tokens placed, subsorted by fewest blocks

    Traverse best path and update tree and deque:
        Create new nodes: leaf (necessarily aligned) or branching
            We learn whether a branching node is actually branching or an unaligned leaf
                only when we try to expand it.
            Eventually we can expand it immediately, so that we'll be able to add three
                types of nodes: aligned leaf, unaligned leaf, and genuine branching.
        Add new nodes to tree
        Add new branching nodes to tail of deque

    Remove head of deque after processing

    No return because graph and deque are both modified in place
    """

    _sa = create_suffix_array(_token_array)
    _frequent_sequences = create_blocks(_sa,_token_membership_array, _witness_count)
    _largest_blocks = find_longest_sequences(_frequent_sequences, _sa)
    if not _largest_blocks: # no blocks, so change type to leaf and remove from queue
        _parent_id = _node_ids.popleft()
        _graph.nodes[_parent_id]["type"] = "leaf"
    else:
        _block_offsets_by_witness, _witness_offsets_to_blocks, _first_token_offset_in_block_by_witness, \
            _first_absolute_token_by_witness, _score_by_block = \
            prepare_for_beam_search(_witness_count, _token_membership_array, _largest_blocks)
        _finished = perform_beam_search(_witness_count, _largest_blocks, _block_offsets_by_witness,
                                        _witness_offsets_to_blocks, _score_by_block)
        # print(f"{_largest_blocks=}")
        # TODO: Only if current head of queue has blocks
        # Get information about parent
        _parent_id = _node_ids.popleft()
        _parent = _graph.nodes[_parent_id] # dictionary of properties
        # Start range for leading unaligned tokens (if any) is start of parent
        _preceding_ends = [i[0] for i in _parent["token_ranges"]]
        # print("Finished: ", _finished)
        # print(f"{_parent['token_ranges']=}")

        # Add blocks as leaf node children (do not add leaf nodes to queue)
        # Precede with potential blocks if there are unaligned preceding tokens
        for _block_id in _finished[0].path[::-1]:
            # _largest_blocks[block] is a leaf node with shape (26, [4, 12795, 25646, 38708, 52026, 66257])
            # The first value is the length of the block (exclusive)
            # The second is the start positions of the block in each witness, using global token position
            _block = _largest_blocks[_block_id] # local offsets
            # ###
            # FIXME: Remove conversion for blocks (which currently works, but needs to be moved here)
            # FIXME: Remove conversion for pre-block unaligned tokens (currently broken)
            # ###
            # print(f"{_block=}")
            if _parent_id == 0: # don't adjust for root
                _adjusted_coordinates = _block[1]
            else:
                _adjusted_coordinates = [i + j[0] for i, j in zip(_block[1], _parent['token_ranges'])]
            # print("Adjusted coordinates: ", _adjusted_coordinates)
            # ###
            # Add potential block first
            # ###
            _current_starts = _adjusted_coordinates # global coordinates
            _parent_starts = [i[0] for i in _parent["token_ranges"]]
            # print(f"{_current_starts=}")
            # print(f"{_parent_starts=}")
            # print(f"{_preceding_ends=}")
            # print([_current_starts[i] + _parent_starts[i] for i in range(_witness_count)])
            # print([_preceding_ends[i] + _current_starts[i] for i in range(_witness_count)])
            if _current_starts != _preceding_ends:
                _id = _graph.number_of_nodes()
                # expand zip for legibility
                _token_ranges = list(zip(_preceding_ends, _current_starts))
                _graph.add_node(_id, type="potential", token_ranges=_token_ranges, children=[])
                _parent["children"].append(_id)
                _node_ids.append(_id)
            # ###
            # Now add block as aligned leaf node
            # Token range is local range + start positions of parent range through same plus block length
            # ###
            _id = _graph.number_of_nodes()
            _token_ranges = [
                (_current_starts[i] + _parent_starts[i], _current_starts[i] + _parent_starts[i] + _block[0])
                for i in range(_witness_count)]
            _graph.add_node(_id, type="leaf", token_ranges=_token_ranges, children=[])
            _parent["children"].append(_id)
            # reset _preceding_ends for loop
            _preceding_ends = [i + _block[0] for i in _adjusted_coordinates]
        # Add trailing unaligned tokens (if any)
        _parent_ends = [i[1] for i in _parent["token_ranges"]]
        if _parent_ends != _preceding_ends:
            # TODO: Process, don't just announce
            print("Need to process trailing tokens")
        else:
            print("No trailing tokens")
        # Reset _parent type property to branching
        _parent["type"] = "branching"
        # Debug report
        # print('Node count: ', len(_graph.nodes))
        # print('Queue size: ', len(_node_ids))


def visualize_graph(_graph: nx.DiGraph):
    # Visualize the tree
    # Create digraph and add root node
    tree = graphviz.Digraph(format="svg")
    # Add all nodes
    # ###
    # RESUME HERE
    # FIXME: Store networkX edges as proper edges (currently property of node)
    # TODO: Add leaf nodes with tokens
    # FIXME YET AGAIN: ranges are wrong and will overrun token array)
    # TODO: Add branching nodes
    # TODO: Add edges (filter out target potential nodes because we aren't adding them)
    # ###
    for node, properties in _graph.nodes(data=True):
        if properties["type"] != "potential":
            print(node, properties)
    # def populate_tree(_digraph, _parent):  # void
    #     for n in _parent.children:
    #         # print(n.id)
    #         if isinstance(n, tuple):  # TODO: remove tuples after creating child nodes
    #             continue
    #         elif isinstance(n, Leaf_node):
    #             label = n.string
    #         else:
    #             label = repr(n.children)
    #         _digraph.node(str(n.id), label=label)
    #         _digraph.edge(str(_parent.id), str(n.id))
    #
    # populate_tree(tree, root)
    # for child in root.children:
    #     if isinstance(child, Branching_node) and child.processed == False:
    #         populate_tree(tree, child)
    # svg_tree = tree.render()
    # display(SVG(svg_tree))