# Tests for create_blocks.py beam-search-related functions
from reptilian import map_blocks_to_graph
import networkx as nx

def test_map_blocks_to_graph():
    assert map_blocks_to_graph.map_blocks_to_graph().number_of_nodes() == 2