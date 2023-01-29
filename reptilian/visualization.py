import graphviz
import networkx as nx


def visualize_graph(_graph: nx.DiGraph, _token_array: list):
    # Visualize the tree including branching nodes
    # Create digraph
    tree = graphviz.Digraph(format="svg")
    # Add all nodes
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
    # Add all edges
    for source, target, properties in _graph.edges(data=True):
        tree.edge(str(source), str(target))
    tree.render("with_branches.gv")  # override automatic filename


def visualize_graph_no_branching_nodes(_graph: nx.DiGraph, _token_array: list):
    # Visualize the tree without branching nodes
    # Order of nodes is depth-first traversal of networkx digraph, which
    #   corresponds to witness order
    # Create root specially, since it's a branching node and would otherwise be skipped
    tree = graphviz.Digraph(format="svg")
    tree.node("root", "ROOT")
    preorder = nx.dfs_preorder_nodes(_graph)
    for _node in preorder:  # id number of node, access as G.nodes[n]
        _properties = _graph.nodes[_node]
        _type = _properties["type"]
        _token_ranges = _properties["token_ranges"]
        # Types are aligned and unaligned
        match _type:
            case 'aligned':
                _tokens = " ".join(_token_array[_token_ranges[0][0]: _token_ranges[0][1]])
                tree.node(str(_node), "\n".join([str(_node), _tokens]))
            case 'unaligned':
                _unaligned_ranges = []
                for i, j in _token_ranges:
                    _unaligned_ranges.append(" ".join(_token_array[i: j]))
                _tokens = "\n".join(_unaligned_ranges)
                tree.node(str(_node), "\n".join([str(_node)+" unaligned", _tokens]))
            case 'branching':
                continue
            case _:
                raise Exception("Unexpected node type: " + _graph[_node]["type"])
        tree.edge("root", str(_node))  # All nodes are children of the root
    tree.render("no_branches.gv")


def visualize_table(_graph: nx.DiGraph, _token_array: list, _witness_count: int):
    # Create table top and bottom
    _table_top = """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE html>
        <html xmlns="http://www.w3.org/1999/xhtml">
            <head>
                <style type="text/css">
                    table, tr, th, td {border: 1px solid black; border-collapse: collapse;}
                    th, td {padding: 3px;}
                    tr:not(:first-child) > th {
                        text-align: right;
                        font-size: smaller;
                        color: gray;
                    }
                    .aligned > td {background-color: #dcdcdc;}
                    .unaligned > td {background-color: beige;}
                </style></head><body><table><tr style="background-color: pink;"><th>Row</th><th>Node</th>
        """ + '\n'.join(
        ['<th style="border: 1px black solid; border-collapse: collapse; text-align: center;">w' + str(i) + '</th>' for
         i in range(_witness_count)]) + '</tr>'
    _table_bottom = '</table></body></html>'

    # Create data rows
    _data_rows = []
    _row_number = -1
    preorder = nx.dfs_preorder_nodes(_graph)
    for _node in preorder:  # id number of node, access as G.nodes[n]
        _properties = _graph.nodes[_node]
        _type = _properties["type"]
        _token_ranges = _properties["token_ranges"]
        # Types are aligned and unaligned
        match _type:
            case 'aligned':
                _row_number += 1
                _tokens = " ".join(_token_array[_token_ranges[0][0]: _token_ranges[0][1]])
                _new_row = "".join(['<tr class="aligned"><th>' + str(_row_number) + '</th><th>', str(_node),
                                    '</th><td colspan=' + str(_witness_count) + '>' + _tokens + '</td></tr>'])
            case 'unaligned':
                _row_number += 1
                _unaligned_ranges = []
                for i, j in _token_ranges:
                    _unaligned_ranges.append(" ".join(_token_array[i: j]))
                _new_data_cells = '\n'.join(['<td>' + i + '</td>' for i in _unaligned_ranges])
                _new_row = "".join(['<tr class="unaligned"><th>' + str(_row_number) + '</th><th>' + str(_node)
                                    + '</th>' + _new_data_cells + '</tr>'])
            case 'branching':
                continue
            case _:
                raise Exception("Unexpected node type: " + _graph[_node]["type"])
        _data_rows.append(_new_row)
    # Compose table (top, data rows, bottom) and write to disk
    table = _table_top + '\n'.join(_data_rows) + _table_bottom
    with open('table-output.html', 'w') as f:
        f.write(table)
