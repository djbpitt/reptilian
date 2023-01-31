# Tests for create_blocks.py beam-search-related functions
from reptilian import map_blocks_to_graph


def test_map_blocks_to_graph():
    # START, END, 8 in first list, 1 duplicate and 1 new in second
    data = [[(36, (0, 113, 229)),
            (1, (38, 152, 268)),
            (25, (41, 153, 269)),
            (7, (67, 179, 295)),
            (6, (74, 187, 303)),
            (9, (81, 194, 310)),
            (22, (90, 204)),
            (2, (90, 204, 320))],
            [(36, (0, 113, 229)),
             (100, (100, 100, 100))]]
    assert map_blocks_to_graph.map_blocks_to_graph(data).number_of_nodes() == 11
