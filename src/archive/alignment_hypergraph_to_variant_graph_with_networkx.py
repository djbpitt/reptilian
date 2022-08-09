#!/usr/bin/env python
# coding: utf-8

# # Alignment hypergraph to variant graph with networkx
# 
# Convert alignment hypergraph to variant graph before visualization.
# 
# (Alignment hypergraph creates a very large dot object, which frustrates visualization. Converting it to a variant graph before visualization reduces the size substantially, at least in normal cases.)

# # FIXME
# 
# A cycle appears in block 597

# ## Initialize

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('load_ext', 'memory_profiler')

import warnings
warnings.filterwarnings('ignore')

import gc # garbage collection
import operator # comparison operators lt and gt
from collections import defaultdict, deque
from typing import Set, List
from dataclasses import dataclass
import networkx as nx
import re
import time

import graphviz
from IPython.display import SVG

import pprint
pp = pprint.PrettyPrinter(indent=2)


# ## Load data

# In[2]:


get_ipython().run_line_magic('run', 'create-blocks.ipynb # data will be in prioritized_blocks')
print('aligning', how_many_paragraphs, 'paraagraphs') # confirm


# ## Create data structures
# 
# 1. Create witness_node (dataclass) for witness nodes. Properties (not all may be needed):
#     1. token string: str
#     1. witness: str
#     1. offset in witness: int
#     1. hyperedge: hyperedge instance (initially null)
# 1. Create alignment_hyperedge (dataclass) for … wait for it! … alignment hyperedges! Properties:
#     1. witness_nodes: set of witness_node objects
# 1. Create lists for witness_node and alignment_hyperedge instances
# 1. Create START and END nodes and associated hyperedges

# In[3]:


@dataclass(unsafe_hash=True)
class WitnessNode:
    """Write a docstring some day"""
    token_string: str
    witness: str
    witness_offset: int
    hyperedge: hyperedge = None

    def __repr__(self):
        return self.token_string


# In[4]:


class AlignmentHyperedge():
    """Write a real docstring someday"""
    def __init__(self, kwargs):
        self.__dict__.update(kwargs) # overwrite the real value
        self._sigla = [key for key in kwargs.keys()]
    def __repr__(self):
        return "|".join(
            [":".join(
                    [str(key), str(getattr(self, key))]
                ) for key in self._sigla]
        )
    def __setitem__(self, key, value):
        if key in self._sigla:
            return
            #raise Exception("Duplicate key exception: "+str(key)+" on object: "+str(self))
        self._sigla.append(key)
        self.__dict__[key] = value
    def __getitem__(self, key):
        return self.__dict__[key]
    def __contains__(self, key):
        return key in self.__dict__
    def values(self):
        return [self[key] for key in self._sigla]
    def items(self):
        return [(key, self.__dict__[key]) for key in self._sigla]


# In[5]:


# Store all witness nodes in dictionary of lists
witness_node_lists = defaultdict(list) # one per witness


# In[6]:


# create START node, nodes for real tokens, and END node
for index, witness_siglum in enumerate(witness_sigla):
    witness_node_lists[witness_siglum].append(WitnessNode('START', witness_siglum, -1, None))
    for witness_token_offset, witness_token in enumerate(witnesses[index]): # list of tokens in a single witness
        witness_node_lists[witness_siglum].append(WitnessNode(witness_token, witness_siglum, witness_token_offset, None))
    witness_node_lists[witness_siglum].append(WitnessNode('END', witness_siglum, len(token_array), None))


# In[7]:


# create hyperedges for START and END
hyperedges = []
offsets = [0, -1]
for offset in offsets:
    dictionary = {siglum: witness_node_lists[siglum][offset] for siglum in witness_sigla}
    hyperedge = AlignmentHyperedge(dictionary)
    for witness_node in dictionary.values():
        witness_node.hyperedge = hyperedge
    hyperedges.append(hyperedge)


# In[8]:


def create_hyperedges(block):
    """Returns hyperedge data for block, together with offset_start_positions"""
    new_hyperedges = []
    for token_offset_in_block in range(selected_block.token_count):
        # first add the token_offset_in_block to the block start positions
        offset_start_positions = [value + token_offset_in_block for value in selected_block.all_start_positions]
        tokens_to_place = [
            (
                witness_sigla[token_membership_array[token_array_index]],
                token_witness_offset_array[token_array_index]
            )
         for token_array_index in offset_start_positions
#         if not placed_token_bitarray[token_array_index] # already placed, so don’t place it again
        ]
        dictionary = { # data for new hyperedge
            siglum: witness_node_lists[siglum][offset + 1]
            for siglum, offset
            in tokens_to_place
        }
        hyperedge = AlignmentHyperedge(dictionary)
        new_hyperedges.append(hyperedge)

    return (offset_start_positions, new_hyperedges)


# In[9]:


def find_floor(siglum, new_offset):
    """Closest hyperedge to the left for a single witness"""
#     print("new_offset: "+ str(new_offset))
    for offset in range(new_offset - 1, -1, -1):
#         print("traversing: "+str(offset))
        # walk backwards looking for node with hyperedge
        node = witness_node_lists[siglum][offset]
#         print("Witness Node: "+str(node))
        if node.hyperedge:
#             print("hyperedge: "+str(node.hyperedge))
            return witness_node_lists[siglum][offset].hyperedge
    raise Exception("We should not get here!")


# In[10]:


def find_ceiling(siglum, new_offset):
    """Closest hyperedge to the right for a single witness"""
    for offset in range(new_offset + 1, len(token_array)):
        # walk forwards looking for node with hyperedge
        if witness_node_lists[siglum][offset].hyperedge:
            return witness_node_lists[siglum][offset].hyperedge
    raise Exception("We should not get here!")


# ## Prose statement of transposition detection
# 
# 1. Find floor and ceiling for block, where both floor and ceiling are sets of hyperedges, with each member of the set selected on the basis of just one witness.
# 1. For each hyperedge in the floor, verify that all nodes that belong to that hyperedge are to the left of the leftmost node for that witness in the block we are trying to place. If so, there is no transposition. If that is not the case, at least one witness would be transposed, so do nothing.
# 1. For each hyperedge in the ceiling, verify that all nodes that belong to that hyperedge are to the right of th rightmost node for that witness in the block we are trying to place. If so, there is no transposition. If that is not the case, at least one witness would be transposed, so do nothing.

# In[11]:


def check_for_crossing_hyperedges(
    hyperedge_to_test: AlignmentHyperedge,
    boundary_hyperedges: List[AlignmentHyperedge], # upper or lower
    lower_or_higher: str) -> bool: # last parameter must be 'floor' or 'ceiling'
    """ Check inside floor and ceiling for consistency

    * Is every floor hyperedge to the left of every leftmost block node?
    * And similarly for the ceiling and the right edge?

    Same function checks 'floor' and 'ceiling' (raise on wrong argument)
    Return True iff no crossing hyperedges, i.e., no transposition
    Relies on operator module for gt and lt; syntax is operator.lt(a, b)

    """
    # Set comparison operator to lt or gt
    if lower_or_higher == 'floor':
        comp = operator.lt
    elif lower_or_higher == "ceiling":
        comp = operator.gt
    else:
        raise Exception("Third argument to check_for_crossing_hyperedges() must be 'floor' or 'ceiling'")

    for siglum, witness_node in hyperedge_to_test.items():
        for boundary_hyperedge in boundary_hyperedges:
            if siglum not in boundary_hyperedge._sigla:
                continue
            elif not comp(boundary_hyperedge[siglum].witness_offset, witness_node.witness_offset):
                return False;
    return True # if we're still alive at the end of the function


# In[12]:


def check_for_transposition(hyperedges_to_check: List[AlignmentHyperedge], debug=None) -> bool:
    """Check for transposition

    List of hyperedges is consecutive because tokens are consecutive in blocks
    Floor of first and ceiling of last are boundaries for the entire list

    Shows debug information for block 597, which introduces transposition
    """
    first_hyperedge = hyperedges_to_check[0]
    last_hyperedge = hyperedges_to_check[-1]
    # print("\n")
    # print(f"{first_hyperedge=}")
    # print(f"{last_hyperedge=}")
    # Find first hyperedge to left for each witness
    #   and determine whether there is a transposition.
    #   They may not be the same hyperedge (!)
    lower_hyperedges = set()
    for siglum, witness_node in first_hyperedge.items():
        # position of node for each witness in first_hyperedge
        # convert from token offset to node offset
        lower_hyperedges.add(find_floor(siglum, witness_node.witness_offset +1))
    # Find ceiling for all witnesses in second hyperedge
    upper_hyperedges = set()
    for siglum, witness_node in last_hyperedge.items():
        # position of node for each witness in last_hyperedge
        # convert from token offset to node offset
        upper_hyperedges.add(find_ceiling(siglum, witness_node.witness_offset +1))
    # we now have: lower_hyperedges, upper_hyperedges
    # Check inside floor and ceiling for consistency; is every floor hyperedge to the left
    #   of every leftmost block node, and similarly for the ceiling and the right edge?

    # If debug is on (block 597, which introduces transposition), show lower and upper hyperedges
    if debug:
        print('Lower hyperedges:')
        pp.pprint(lower_hyperedges)
        print('Upper hyperedges:')
        pp.pprint(upper_hyperedges)
    # End of debug check for block 597
    floor_ok = check_for_crossing_hyperedges(first_hyperedge, lower_hyperedges, 'floor')
    ceiling_ok = check_for_crossing_hyperedges(last_hyperedge, upper_hyperedges, 'ceiling')
    return floor_ok and ceiling_ok # returns true only if both edges show no transposition


# ## Augmenting an existing hyperedge
# 
# * Every witness node always contains zero or one hyperedge. If zero, create new. If one, extend.
# * Provisionally, check for two or more and raise an exception. Our assumption (okay, hope) is that it won't happen.

# In[13]:


def merge_hyperedge(potential_hyperedge):
            # check whether a witness_node has got the hyperedge property set
            # a hyperedge may contain zero, some, or all witnesses; we need to handle all these cases
            found = set([witnessnode.hyperedge for siglum, witnessnode in potential_hyperedge.items() if witnessnode.hyperedge])
            if len(found) > 1:
                # found two or more existing hyperedges...
                # remove them from the hyperedges set
                for existing_hyperedge in found:
                    hyperedges.remove(existing_hyperedge)
                # replace them by a new hyperedge
                hyperedges.append(potential_hyperedge)
                # update hyperedge property on the witness nodes involved in the hyperedge
                new_hyperedge = potential_hyperedge
                for witness_node in new_hyperedge.values():
                    witness_node.hyperedge = new_hyperedge

#                 print(found) # debug only
#                 raise Exception('Eek! Nodes belong to different existing hyperedges!')
            elif len(found) == 0:
                # we checked for transpositions and we filtered out hyperedges for witness nodes that are already placed
                hyperedges.append(potential_hyperedge)
                # update hyperedge property on the witness nodes involved in the hyperedge
                new_hyperedge = potential_hyperedge
                for witness_node in new_hyperedge.values():
                    witness_node.hyperedge = new_hyperedge
            else: # update single existing hyperedge
                for siglum in potential_hyperedge._sigla:
                    list(found)[0][siglum] = potential_hyperedge[siglum] # add node to hyperedge (possibly redundantly)
                    potential_hyperedge[siglum].hyperedge = list(found)[0] # and update hyperedge property of node


# In[14]:


# debug only
# def map_alignment_graph_to_networkx():
#     AG = nx.DiGraph()

#     # create nodes and regular edges (start and end are regular witness nodes in alignment graph)
#     for siglum, witness_node_list in witness_node_lists.items():
#         for index,witness_node in enumerate(witness_node_list):
#             AG.add_node(witness_node)
#             if index > 0:
#                 AG.add_edge(witness_node_list[index - 1], witness_node)

#     # add hyperedges (fake: connect only closest witnesses, pairwise)
#     for hyperedge in hyperedges: # find all sigla, in order, on hyperedge
#         hyperedge_sigla = hyperedge._sigla
#         filtered_sigla = list(filter(lambda x: x in hyperedge_sigla, witness_sigla)) # sigla in hyperedge, in order
#         for source_siglum, target_siglum in zip(filtered_sigla, filtered_sigla[1:]): # sigla, not nodes!
#             source_node = hyperedge[source_siglum]
#             target_node = hyperedge[target_siglum]
#             AG.add_edge(source_node, target_node)

#     # we've added all nodes and edges correctly(!), so return networkx graph to test for cycles
#     return AG


# In[15]:


class VG_node():
    """Variant graph node"""
    def __init__(self, token_string, data): # dictionary of siglum:witness_offset
        self.token_string = token_string
        self.__dict__.update(**data)
        self._sigla = [key for key in data.keys()]
    def __repr__(self):
        return self.token_string + "~" + "|".join([":".join([str(key), str(getattr(self, key))]) for key in self.sigla()])
    def __setitem__(self, key, value):
        self._sigla.append(key)
        self.__dict__[key] = value
    def __getitem__(self, key):
        return self.__dict__[key]
    def __contains__(self, key):
        return key in self.__dict__
    def sigla(self):
        return self._sigla
    def values(self):
        return [self[key] for key in self._sigla]
    def items(self):
        return [(key, self.__dict__[key]) for key in self._sigla]


# In[16]:


# Create networkx variant graph from our (not networkx) alignment graph
# Create variant graph edges for start and end nodes (only)
def augment_or_add_edge_without_conversion(siglum, source_VG_node, target_VG_node):
    if VG.has_edge(source_VG_node, target_VG_node):
        VG[source_VG_node][target_VG_node]["siglum"].append(siglum)
    else:
        VG.add_edge(
            source_VG_node,
            target_VG_node,
            siglum = [siglum]
        )

# Use alignment graph node to look up variant graph node
def alignment_node_to_VG_node(alignment_node: WitnessNode) -> VG_node:
    global witness_offset_to_VG_node
    return witness_offset_to_VG_node[alignment_node.witness][alignment_node.witness_offset]

# Create variant graph data edges (except start and end nodes)
def augment_or_add_edge(siglum, source, target):
    source_VG_node = alignment_node_to_VG_node(source)
    target_VG_node = alignment_node_to_VG_node(target)
    if VG.has_edge(source_VG_node, target_VG_node):
        VG[source_VG_node][target_VG_node]["siglum"].append(siglum)
    else:
        VG.add_edge(
            source_VG_node,
            target_VG_node,
            siglum = [siglum]
                )

def create_variant_graph_from_alignment_graph():
    # create variant graph and add start and end nodes

    start_node = VG_node("START", {})
    end_node = VG_node("END", {})
    global VG
    VG = nx.DiGraph(start = start_node, end = end_node)
    VG.add_node(start_node)
    VG.add_node(end_node)

    # keep track of which witness nodes (and therefore also hyperedges) 
    # have been added to VG
    from bitarray import bitarray
    global VG_tracking
    VG_tracking = {}
    for siglum, witness_node_list in witness_node_lists.items():
        VG_tracking[siglum] = bitarray(len(witness_node_list) - 2)
        VG_tracking[siglum].setall(0)

    # map from witness node to variant graph node (needed to construct edges)
    # values will be added as we create variant graph nodes
    global witness_offset_to_VG_node
    witness_offset_to_VG_node = {}
    for siglum in witness_node_lists.keys():
        witness_offset_to_VG_node[siglum] = [None] * (len(witness_node_lists[siglum]) - 2)

    # Create variant graph nodes
    # witness_node_lists maps from siglum to list of witness node objects
    for siglum, witness_node_list in witness_node_lists.items(): # witness by witness
        for witness_node in witness_node_list:
            if not witness_node.token_string in ("START", "END"): # create separately
                if not witness_node.hyperedge:
                    # If it doesn't have a hyperedge, create a new variant graph node and add data from witness node
                    new_VG_node = VG_node(
                            witness_node.token_string,
                            {witness_node.witness: witness_node.witness_offset}
                        )
                    VG.add_node(new_VG_node)
                    VG_tracking[siglum][witness_node.witness_offset] = 1 # update bitarray to show that witness node has been added
                    witness_offset_to_VG_node[siglum][witness_node.witness_offset] = new_VG_node # add pointer from witness note to VG node
                elif VG_tracking[siglum][witness_node.witness_offset] == 1:
                    # If it does have a hyperedge and the hyperedge has already been created, do nothing
                    continue
                else:
                    # Otherwise it has a new hyperedge, so:
                    # 1. create it with empty dictionary
                    # 2. add witness nodes from hyperedge nodes to new VG node dictionary
                    # 3. update bitarray
                    # 4. update pointers from witness node to new VG node
                    new_VG_node = VG_node(
                            witness_node.token_string,
                            {}
                        )
                    VG.add_node(new_VG_node)
                    for hyperedge_node in witness_node.hyperedge.values():
                        new_VG_node[hyperedge_node.witness] = hyperedge_node.witness_offset
                        VG_tracking[hyperedge_node.witness][hyperedge_node.witness_offset] = 1
                        witness_offset_to_VG_node[hyperedge_node.witness][hyperedge_node.witness_offset] = new_VG_node

    for siglum, witness_node_list in witness_node_lists.items():
        # Add start and end edges
        augment_or_add_edge_without_conversion(
            siglum,
            VG.graph["start"],
            witness_offset_to_VG_node[siglum][witness_node_list[1].witness_offset]
        )
        augment_or_add_edge_without_conversion(
            siglum, 
            witness_offset_to_VG_node[siglum][witness_node_list[-2].witness_offset], 
            VG.graph["end"]
        )
        # Congratulations! You've added start and end edges! Now add data edges
        edge_pairs = zip(witness_node_list[1:-2], witness_node_list[2:]) # START and END already created
        for source, target in edge_pairs:
            augment_or_add_edge(siglum, source, target)

    return VG


# In[17]:


# create all hyperedges (gingerly)
selected_blocks = filter(lambda x: len(x.all_start_positions) == x.witness_count, prioritized_blocks)
# For debugging process only some of the selected blocks
for index, selected_block in enumerate(selected_blocks):
    print("Processing block #" + str(index))
    # %memit pass
    offset_start_positions, new_hyperedges = create_hyperedges(selected_block)
    # Set debug flag for block 597 and pass to transposition check
    debug_flag = True if index == 597 else False
    no_transposition = check_for_transposition(new_hyperedges, debug_flag)
    if no_transposition: # all-or-nothing
        for potential_hyperedge in new_hyperedges:
            merge_hyperedge(potential_hyperedge)
    # Debug only: cycle detection
    # Map our alignment graph in our code onto networkx variant graph
    # Check for cycles, break and report if cycle is found
    variant_graph = create_variant_graph_from_alignment_graph()
    try:
        nx.algorithms.cycles.find_cycle(variant_graph)
        break
    except nx.exception.NetworkXNoCycle:
        continue
print(f"There are {len(list(filter(lambda x: len(x.all_start_positions) == x.witness_count, prioritized_blocks)))=} blocks without repetition and we processed {index} of them")


# ## Reduce memory use

# In[18]:


print(f"{len(prioritized_blocks)=}")
print(f"{prioritized_blocks[0].token_count=}")
del selected_blocks
del prioritized_blocks
del raw_data_dict
del lcp_array
del suffix_array
del token_array
del token_membership_array
del token_to_block_dict
del token_witness_offset_array
del witnesses
gc.collect()


# ## Visualize original alignment graph

# In[19]:


# node id values must be strings for graphviz
a = graphviz.Digraph(format="svg", name="alignment_graph")
a.attr(rankdir = "TB")
a.attr(rank = 'same')
a.attr(compound='true')

# create nodes and regular edges
for siglum, witness_node_list in witness_node_lists.items():
    c = graphviz.Digraph(siglum+'child') # can't reuse subgraph label, so prepend siglum
    c.attr(rankdir = "LR")
    c.attr(rank='same')
    for index,witness_node in enumerate(witness_node_list):
        node_id = "#".join((witness_node.witness, str(witness_node.witness_offset)))
        preceding_node_id = "#".join((witness_node_list[index - 1].witness, str(witness_node_list[index - 1].witness_offset)))
        c.node(node_id, "".join((node_id, '\n', witness_node.token_string)))
        if witness_node.token_string != 'START':
            c.edge(preceding_node_id, node_id)
    a.subgraph(c)

# add hyperedges (fake: connect only closest witnesses, pairwise)
for hyperedge in hyperedges: # find all sigla, in order, on hyperedge
    hyperedge_sigla = hyperedge._sigla
    filtered_sigla = list(filter(lambda x: x in hyperedge_sigla, witness_sigla)) # sigla in hyperedge, in order
    # print(hyperedge)
    for source_siglum, target_siglum in zip(filtered_sigla, filtered_sigla[1:]): # sigla, not nodes!
        source_node = hyperedge[source_siglum]
        target_node = hyperedge[target_siglum]
        if source_node.token_string in ("START", "END"): # force alignment by weighting start and end edges
            edge_weight="100"
        else:
            edge_weight="1"
        a.edge(
            "#".join((source_siglum, str(source_node.witness_offset))),
            "#".join((target_siglum, str(target_node.witness_offset))),
            weight=edge_weight
              )
SVG(a.view())


# In[20]:


# Debug code: do we get here?
print('We have created and rendered an alignment graph. Yay us!')


# ## Create variant graph from alignment hypergraph
# 
# First create variant graph nodes. Walk over witness nodes in `witness_node_lists`. Keep track of which hyperedges have been added to variant graph. For each witness node:
# 
# 1. If node has no hyperedge, create variant graph node with selected information from witness node on it.
# 1. If node has new hyperedge, create variant graph node with selected information from all witness nodes from hyperedge on it.
# 1. If node has hyperedge that is already on variant graph, do nothing.
# 
# Witness nodes contain token string, a witness siglum, a witness offset, and an optional hyperedge. The hyperedge contains, as dictionary items, all witness sigla as keys with witness nodes as values. From this information, we put onto the variant graph node the token string plus the hyperedge dictionary with sigla as keys and witness offsets as values.
# 
# **Notes:**
# 
# * The variant graph does not need to contain the token string, since we can look it up later, but for ease of visualization we include it here.
# * In Real Life Collatex a variant graph node contains properties other than the token string. We ignore that structure in our experiment as we concentrate on visualization.

# In[21]:


# # recover some memory
del witness_node_list
del witness_offset_to_VG_node
del VG_tracking
gc.collect()


# ## Visualize variant graph

# In[22]:


# # node id values must be strings for graphviz
a = graphviz.Digraph(format="svg", name="variant_graph_unjoined")
a.attr(rankdir = "LR")
a.attr(rank = 'same')
a.attr(compound='true')

# plot nodes, building {node: id} for lookup
node_to_id = {}
for index, node in enumerate(VG.nodes()):
    node_id = str(index)
    node_to_id[node] = node_id
    node_text = node.token_string + " (" + node_id + ")"
    a.node(node_id, label=node_text)

# plot edges
for edge in VG.edges(data=True):
    # edge is a three-item tuple: source, target, dictionary of properties
    label = "(all)" if len(edge[2]["siglum"]) == len(witness_sigla) else ",".join(sorted(edge[2]["siglum"]))
    a.edge(node_to_id[edge[0]], node_to_id[edge[1]], label=label)

# print('aligning', how_many_paragraphs, 'paragraphs') # confirm
SVG(a.view())


# In[23]:


# #  This function joins the variant graph in place.
# #  This function is a straight port of the Java version of CollateX.

# def join(graph):
#     processed = set()
#     end = graph.graph["end"]
#     queue = deque()
#     for (_, neighbor) in graph.out_edges(graph.graph["start"]):
#         queue.appendleft(neighbor)
#     while queue:
#         vertex = queue.popleft()
#         out_edges = graph.out_edges(vertex)
#         if len(out_edges) == 1:
#             (_, join_candidate) = next(iter(out_edges))
#             can_join = join_candidate != end and len(graph.in_edges(join_candidate)) == 1
#             if can_join:
#                 join_vertex_and_join_candidate(graph, join_candidate, vertex)
#                 for (_, neighbor, data) in list(graph.out_edges(join_candidate, data=True)):
#                     graph.remove_edge(join_candidate, neighbor)
#                     if graph.has_edge(vertex, neighbor):
#                         graph[source][target]["siglum"].append(data["siglum"])
#                     else:
#                         graph.add_edge(vertex, neighbor, siglum=data["siglum"])
#                 graph.remove_edge(vertex, join_candidate)
#                 graph.remove_node(join_candidate)
#                 queue.appendleft(vertex)
#                 continue
#         processed.add(vertex)
#         for (_, neighbor) in out_edges:
#             # FIXME: Why do we run out of memory in some cases here, if this is not checked?
#             if neighbor not in processed:
#                 queue.appendleft(neighbor)


# def join_vertex_and_join_candidate(graph, join_candidate, vertex):
#     # Note: since there is no normalized/non normalized content in the graph
#     # a space character is added here for non punctuation tokens

#     if re.match(r'^\W', join_candidate.token_string):
#         vertex.token_string += join_candidate.token_string
#     else:
#         vertex.token_string += (" " + join_candidate.token_string)
#     # join_candidate must have exactly one token (inside a list); left item may have more
# #     for siglum, token in join_candidate.tokens.items():
# #         vertex.add_token(siglum, token[0])


# In[24]:


# len(nx.algorithms.cycles.find_cycle(VG))


# In[25]:


# nx.algorithms.cycles.find_cycle(VG)


# In[26]:


# join(VG)


# In[27]:


# pp.pprint([edge for edge in VG.edges()])


# In[28]:


# # node id values must be strings for graphviz
# a = graphviz.Digraph(format="svg", name="variant_graph_joined")
# a.attr(rankdir = "LR")
# a.attr(rank = 'same')
# a.attr(compound='true')

# # plot nodes, building {node: id} for lookup
# node_to_id = {}
# for index, node in enumerate(VG.nodes()):
#     node_id = str(index)
#     node_to_id[node] = node_id
#     node_text = node.token_string + " (" + node_id + ")"
#     a.node(node_id, label=node_text)

# # plot edges
# for edge in VG.edges(data=True):
#     # edge is a three-item tuple: source, target, dictionary of properties
#     label = "(all)" if len(edge[2]["siglum"]) == len(witness_sigla) else ",".join(sorted(edge[2]["siglum"]))
#     a.edge(node_to_id[edge[0]], node_to_id[edge[1]], label=label)

# print('aligning', how_many_paragraphs, 'paragraphs') # confirm
# SVG(a.view())


# In[29]:


# nx.algorithms.cycles.find_cycle(VG)


# # To do next
# 
# 1. Move SVG/Graphviz code into function
# 1. Create alignment table visualization without joining
# 1. Add joining to the alignment table (to check accuracy of joining results)
# 1. Edge labels are wrong, which may be a problem with the join() function
# 1. Test intermediate data sets, larger than a paragraph and smaller than a chapter
# 1. Reassess method of prioritizing blocks
# 1. Implement decision tree / graph

# In[ ]:




