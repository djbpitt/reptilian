import pprint
from collections import deque

from import_witnesses import import_witnesses
# from alignment_tree import create_tree, expand_node # FIXME: functions are now in create_blocks.py, so remove
from create_blocks import create_token_array, create_tree, expand_node
from visualization import *
from export import *

pp = pprint.PrettyPrinter(indent=2)

# ###
# Create full token array and related resources from witness data
# ###
sigla, witnesses = import_witnesses()
token_array, token_membership_array, token_witness_offset_array, token_ranges = create_token_array(witnesses)
# print(f"{len(token_array)=}")

# ###
# Initialize alignment tree and add root
# nodes_to_process is queue of nodes to check for expansion
# (deque for performance reasons; we use only FIFO, so regular queue)
# ###
alignment_tree = create_tree()
alignment_tree.add_node(0, type="potential", token_ranges=token_ranges)
nodes_to_process = deque([0])

# ###
# Expand tree, starting at root
# ###
counter = 0
while nodes_to_process:
    # print('Iteration #', counter)
    counter += 1
    # print("Head of queue: ", alignment_tree.nodes[nodes_to_process[0]]['token_ranges'])
    if counter == 1:  # special handling for root node
        expand_node(alignment_tree,
                    nodes_to_process,
                    token_array,
                    token_membership_array,
                    len(witnesses))
        continue
    # All nodes except root
    local_token_array = []
    local_token_membership_array = []
    for index, token_range in enumerate(alignment_tree.nodes[nodes_to_process[0]]['token_ranges']):
        local_token_array.extend(token_array[token_range[0]: token_range[1]])
        local_token_membership_array.extend(token_membership_array[token_range[0]: token_range[1]])
        if index < len(witnesses) - 1:
            local_token_array.append(' #' + str(index + 1) + ' ')
            local_token_membership_array.append(' #' + str(index + 1) + ' ')
    # print("Local token array: ", local_token_array)
    # print("Local token membership array: ", local_token_membership_array)
    expand_node(alignment_tree,
                nodes_to_process,
                local_token_array,
                local_token_membership_array,
                len(witnesses))
# print('Node count: ', len(alignment_tree.nodes))
# print('Edge count: ', len(alignment_tree.edges))
# print('Queue size: ', len(nodes_to_process))
# print(f"{witnesses=}")
# with open('nodes.txt', 'w') as f:
#     f.write(str(alignment_tree.nodes(data=True)))
# with open('edges.txt', 'w') as f:
#     f.write(str(alignment_tree.edges(data=True)))
# with open('queue.txt', 'w') as f:
#     f.write(str(nodes_to_process))
# ###
# Visualizations write to disk (filenames specified in visualization.py):
#   no_branches.gv.svg
#   with_branches.gv.svg
#   table-output.html
# ###
# visualize_graph(alignment_tree, token_array)
# visualize_graph_no_branching_nodes(alignment_tree, token_array)
# visualize_table(alignment_tree, token_array, len(witnesses))
export_unaligned(alignment_tree, token_array)