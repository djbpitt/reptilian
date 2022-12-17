"""Tree of nodes

Node properties:
    id:int : consecutive integers
    type:str : aligned, unaligned, branching, potential (= unexpended)
    token_ranges:list[Tuple] : offsets in global token array
"""
import networkx as nx
from collections import deque
from typing import List
from create_blocks import *
import graphviz
from IPython.display import SVG, HTML, display

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
    if not _largest_blocks: # no blocks, so change type to unaligned and remove from queue
        _parent_id = _node_ids.popleft()
        _graph.nodes[_parent_id]["type"] = "unaligned"
        return
    _block_offsets_by_witness, _witness_offsets_to_blocks, _first_token_offset_in_block_by_witness, \
        _first_absolute_token_by_witness, _score_by_block = \
        prepare_for_beam_search(_witness_count, _token_membership_array, _largest_blocks)
    _finished = perform_beam_search(_witness_count, _largest_blocks, _block_offsets_by_witness,
                                    _witness_offsets_to_blocks, _score_by_block)
    # print(f"{_largest_blocks=}")
    # Get information about parent
    _parent_id = _node_ids.popleft()
    _parent = _graph.nodes[_parent_id] # dictionary of properties
    # Start range for leading unaligned tokens (if any) is start of parent
    _preceding_ends = [i[0] for i in _parent["token_ranges"]]
    # print("Finished: ", _finished)
    # print(f"{_parent['token_ranges']=}")

    # Add blocks as aligned nodes and add edges from parent to new node (do not add leaf nodes to queue)
    # Precede with potential blocks if there are unaligned preceding tokens
    for _block_id in _finished[0].path[::-1]:
        # _largest_blocks[block] is a leaf node with shape (26, [4, 12795, 25646, 38708, 52026, 66257])
        # The first value is the length of the block (exclusive)
        # The second is the start positions of the block in each witness, using global token position
        _block = _largest_blocks[_block_id] # local offsets
        # print(f"{_block=}")
        # print(f"{_parent['token_ranges']=}")
        if _parent_id == 0: # don't adjust for root
            _adjusted_coordinates = _block[1]
        else:
            _adjusted_coordinates = [i + j[0] - k for i, j, k in zip(_block[1], _parent['token_ranges'],
                                                                     _first_absolute_token_by_witness)]
        # print("Adjusted coordinates: ", _adjusted_coordinates)
        # ###
        # Add potential block first
        # ###
        _current_starts = _adjusted_coordinates # global coordinates
        # print(f"{_current_starts=}")
        # print(f"{_current_starts=}")
        # print(f"{_parent_starts=}")
        # print(f"{_preceding_ends=}")
        # print([_current_starts[i] + _parent_starts[i] for i in range(_witness_count)])
        # print([_preceding_ends[i] + _current_starts[i] for i in range(_witness_count)])
        if _current_starts != _preceding_ends:
            _id = _graph.number_of_nodes()
            # expand zip for legibility
            _token_ranges = list(zip(_preceding_ends, _current_starts))
            _graph.add_node(_id, type="potential", token_ranges=_token_ranges, parent_id=_parent_id)
            _graph.add_edge(_parent_id, _id)
            _node_ids.append(_id)
        # ###
        # Now add block as aligned leaf node
        # Token range is local range + start positions of parent range through same plus block length
        # ###
        _id = _graph.number_of_nodes()
        _token_ranges = [
            (_current_starts[i], _current_starts[i] + _block[0])
            for i in range(_witness_count)]
        _graph.add_node(_id, type="aligned", token_ranges=_token_ranges, parent_id=_parent_id)
        _graph.add_edge(_parent_id, _id)
        # reset _preceding_ends for loop
        _preceding_ends = [i + _block[0] for i in _adjusted_coordinates]
    # Add trailing unaligned tokens (if any)
    _parent_ends = [i[1] for i in _parent["token_ranges"]]
    if _parent_ends != _preceding_ends:
        # print("Need to process trailing tokens")
        # print(f"{_preceding_ends=}")
        # print(f"{_parent_ends=}")
        _token_ranges = list(zip(_preceding_ends, _parent_ends))
        # print(f"{_token_ranges=}")
        _id = _graph.number_of_nodes()
        _graph.add_node(_id, type="potential", token_ranges=_token_ranges, parent_id = _parent_id)
        _graph.add_edge(_parent_id, _id)
        _node_ids.append(_id)
    else:
        print("No trailing tokens")
    # Reset _parent type property to branching
    _parent["type"] = "branching"
    # Debug report
    # print('Node count: ', len(_graph.nodes))
    # print('Queue size: ', len(_node_ids))


def visualize_graph(_graph: nx.DiGraph, _token_array: list):
    # Visualize the tree
    # Create digraph and add root node
    tree = graphviz.Digraph(format="svg")
    # Add all nodes
    # ###
    # RESUME HERE
    # ###
    for node, properties in _graph.nodes(data=True):
    # Types are aligned, unaligned, potential, branching
        match properties["type"]:
            case 'aligned':
                _tokens = " ".join(_token_array[properties["token_ranges"][0][0]: properties["token_ranges"][0][1]])
                tree.node(str(node), "\n".join([str(node), _tokens]))
            case 'unaligned':
                # print("Visualizing node #" + str(node))
                # print(properties["token_ranges"])
                _unaligned_ranges = []
                for i, j in properties["token_ranges"]:
                    # print(" ".join(_token_array[i: j]))
                    _unaligned_ranges.append(" ".join(_token_array[i: j]))
                _tokens = "\n".join(_unaligned_ranges)
                # print(_tokens)
                tree.node(str(node), "\n".join([str(node)+" unaligned", _tokens]))
            case 'potential':
                # Should not appear once alignment is complete
                tree.node(str(node), "POTENTIAL")
            case 'branching':
                tree.node(str(node), "BRANCHING")
            case _:
                raise Exception("Unexpected node type: " + properties["type"])
    for source, target, properties in _graph.edges(data=True):
        tree.edge(str(source), str(target))
    tree.render("with_branches.gv") # saves automatically as Digraph.gv.svg

def visualize_graph_no_branching_nodes(_graph: nx.DiGraph, _token_array: list):
    # Visualize the tree without branching nodes
    # Order of nodes is depth-first traversal of networkx digraph, which
    #   corresponds to witness order
    # All nodes are children of the root, with edges ordered
    preorder = nx.dfs_preorder_nodes(_graph)
    no_branching_nodes = [node for node in preorder if _graph.nodes[node]["type"] != "branching"]
    return no_branching_nodes
