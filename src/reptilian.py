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
alignment_tree.add_node(0, type="branching", token_ranges=token_ranges, expanded=False, children=[])
nodes_to_process = deque([0])

# ###
# Expand tree, starting at root
# ###
frequent_sequences = expand_node(alignment_tree,
            nodes_to_process,
            token_array,
            token_membership_array,
            len(witnesses))


