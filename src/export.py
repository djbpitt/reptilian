import networkx as nx
import json

def export_unaligned(_graph: nx.DiGraph, _token_array: list):
    """Dump unaligned nodes to explore non-full-depth strategies

    Export as JSON array
        [
          {
            nodeno: 1,
            readings: [
              [ witnessA-tokens-as-strings ],
              [ witnessB-tokens-as-strings ]
            ]
          },
          {
            nodeno: 3,
            readings: [
              [ witnessA-tokens-as-strings ],
              [ witnessB-tokens-as-strings ]
            ]
          }
        ]
    """
    unaligned_data = []
    for node, properties in _graph.nodes(data=True):
        # Types are aligned, unaligned, potential, branching
        match properties["type"]:
            case 'aligned':
                continue
            case 'unaligned':
                _node_data = {}
                _node_data["nodeno"] = node
                _readings = []
                for i, j in properties["token_ranges"]:
                    _readings.extend([_token_array[i: j]])
                _node_data["readings"] = _readings
                unaligned_data.append(_node_data)
            case 'branching':
                continue
            case _:
                raise Exception("Unexpected node type: " + properties["type"])
    json_object = json.dumps(unaligned_data, indent=2)
    with open("unaligned_data.json", "w") as f:
        f.write(json_object)

