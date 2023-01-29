# Tests for create_blocks.py beam-search-related functions
from reptilian import create_blocks
import networkx as nx

def test_map_blocks_to_graph():
    assert create_blocks.map_blocks_to_graph().number_of_nodes() == 2