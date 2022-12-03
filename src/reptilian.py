# ###
# External imports
# ###
import graphviz
from IPython.display import display, HTML, SVG
import pprint

pp = pprint.PrettyPrinter(indent=2)
import re
from typing import List

# ###
# Local imports
# ###
from import_witnesses import import_witnesses
from create_blocks import create_token_array
from alignment_tree import *

# ###
# Create full token array and related resources from witness data
# ###
sigla, witnesses = import_witnesses()
token_array, token_membership_array, token_witness_offset_array, token_ranges = create_token_array(witnesses)

# ###
# Initialize alignment tree and add root
# No longer need node_by_id dictionary because
#   networkx builds that in
# nodes_to_process is queue of nodes to check for expansion
# TODO: Constrain cooccurrence of node attributes
# ###
alignment_tree = create_tree()
alignment_tree.add_node(0, type="potential", token_ranges=token_ranges, children=[])
nodes_to_process = deque([0])

# ###
# Expand tree, starting at root
# ###
counter = 0
while nodes_to_process and counter < 10:
    print('Iteration #', counter)
    # TODO: Make it pretty
    # TODO: Compute len(witnesses) just once
    # FIXME: Blocks in tiers after the first are storing local, rather than global, token offsets
    # FIXME: Pre-block unaligned tokens in tiers after the first have incorrect (local?) second range values
    print("Head of queue: ", alignment_tree.nodes[nodes_to_process[0]]['token_ranges'])
    if counter > 0:
        local_token_array = []
        for index, token_range in enumerate(alignment_tree.nodes[nodes_to_process[0]]['token_ranges']):
            local_token_array.extend(token_array[token_range[0]: token_range[1]])
            if index < len(witnesses) - 1:
                local_token_array.append(' #' + str(index + 1) + ' ')
        local_token_membership_array = []
        for index, token_range in enumerate(alignment_tree.nodes[nodes_to_process[0]]['token_ranges']):
            local_token_membership_array.extend(token_membership_array[token_range[0]: token_range[1]])
            if index < len(witnesses) - 1:
                local_token_membership_array.append(' #' + str(index + 1) + ' ')
        # print("Local token array: ", local_token_array)
        # print("Local token membership array: ", local_token_membership_array)
        expand_node(alignment_tree,
                    nodes_to_process,
                    local_token_array,
                    local_token_membership_array,
                    len(witnesses))
    else:
        expand_node(alignment_tree,
                    nodes_to_process,
                    token_array,
                    token_membership_array,
                    len(witnesses))
    counter += 1
print('Node count: ', len(alignment_tree.nodes))
print('Queue size: ', len(nodes_to_process))
with open('nodes.txt', 'w') as f:
    f.write(str(alignment_tree.nodes(data=True)))
with open('queue.txt', 'w') as f:
    f.write(str(nodes_to_process))
# print(f"{witnesses=}")
