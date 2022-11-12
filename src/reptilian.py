# ###
# External imports
# ###
from bisect import bisect_right
import graphviz
from heapq import *  # priority heap, https://docs.python.org/3/library/heapq.html
from IPython.display import display, HTML, SVG
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=2)
import re
from typing import List

# ###
# Local imports
# ###
from import_witnesses import import_witnesses
from create_blocks import create_token_array

# ###
# Main
# ###
sigla, witnesses = import_witnesses()
token_array, token_membership_array, token_witness_offset_array, token_ranges = create_token_array(witnesses)

# 2022-11-12
# Resume by deciding how to create root node (and other nodes)
# Use len(alignment_tree.node_by_id) + 1 to get new id value
# Should alignment_tree be a traditional class with a method to
#   add a new node? Should we consider a monad?
# Do we want to expose node_by_id?
