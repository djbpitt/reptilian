#!/usr/bin/env python
# coding: utf-8

# # This time for sure
# 
# Alignment is modeled as masked array

# ## Initialize

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from collections import defaultdict, deque
from typing import Set, List
from dataclasses import dataclass
import networkx as nx
import re
import queue

import numpy as np
import numba as nb
import numpy.ma as ma

import graphviz
from IPython.display import SVG

from numba import jit

import pprint
pp = pprint.PrettyPrinter(indent=2)
debug = True

get_ipython().run_line_magic('load_ext', 'line_profiler')


# In[2]:


@nb.njit(parallel=True)
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@nb.njit
def np_apply_along_axis_nonparallel(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@nb.njit(parallel=True)
def np_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)

@nb.njit(parallel=True)
def np_std(array, axis):
  return np_apply_along_axis(np.std, axis, array)

@nb.njit # errors out with parallel=True
def np_prod(array, axis):
  return np_apply_along_axis_nonparallel(np.prod, axis, array)


# In[3]:


# RESUME HERE
#
# Issue: numba cannot operate on masked arrays
#
# Possible work-around:
#   separate data and mask
#   separate compiled numpy functions for data and mask
#   reassemble into new masked array
#
# Documentation note:
#
# Matrix = two-dimensional
# Vector = one-dimensional
# We do not use the term "array"

@nb.njit(parallel=True)
def ma_subtract(matrix, vector): #todo
    # subtract vector from each row of matrix
    # input is matrix of numbers, vector of numbers
    # axis does not need to be specified
    # returns matrix of numbers
    return matrix - vector

@nb.njit(parallel=True)
def ma_compute_either_masked(matrix, vector): # ready for testing
    # return True (masked) if either value is masked
    # input is matrix of booleans, vector of booleans
    # axis does not need to be specified
    # returns matrix of booleans
    return matrix | vector

@nb.njit(parallel=True)
def ma_compute_both_masked(matrix, vector): # ready for testing
    # return True (masekd) only if both values are masked
    # input is matrix of booleans, vector of booleans
    # axis does not need to be specified
    # returns matrix of booleans
    return matrix * vector

@nb.njit
def ma_prod(matrix, axis): # ready for testing
    # multiply across dimension
    # input is matrix of numbers
    # axis is 0 or 1
    # return vector of numbers
    return np_prod(matrix, 1)

@nb.njit(parallel=True)
def ma_compare_to_zero(vector): # ready for testing
    # input is vector of numbers (vector) and vector of booleans (mask)
    # evaluate to True if product of non-masked values == 0, otherwise false
    # return vector of booleans
    return vector == 0


# In[4]:


# test of ma_subtract()
#
# Input is matrix[int], vector[int]
# Return is matrix[int]
#
a = np.array([[1,2,3,4,5],
              [1,2,3,4,5],
              [1,2,3,4,5]
             ])
v = np.array([10, 100, 1000, 10000, 100000])
print(a - v)
print(ma_subtract(a, v))


# In[5]:


# test of ma_compute_either_masked()
# return True if either is True
#
# Input is matrix[bool], vector[bool]
# Return is matrix[bool]
#
a = np.array([[True, True, False, False],
              [True, True, True, False],
              [False, True, False, True]
             ])
b = np.array([True, False, True, False])
ma_compute_either_masked(a, b)


# In[6]:


# test of ma_compute_both_masked()
# return True iff both are True
#
# Input is matrix[bool], vector[bool]
# Return is matrix[bool]
#
a = np.array([[True, True, False, False],
              [True, True, True, False],
              [False, True, False, True]
             ])
b = np.array([True, False, True, False])
ma_compute_both_masked(a, b)


# In[7]:


# test of ma_prod()
#
# input is matrix of numbers
# axis must be specified
# return vector of numbers

a = np.array([[1,2,3,4,5],
              [1,3,5,2,2],
              [3,2,1,2,3]
             ])
ma_prod(a, 1)


# In[8]:


# test of ma_compare_to_zero()
# input is vector of numbers
# return vector of booleans
#
a = np.array([1,2,3,0,4,5,0,6])
ma_compare_to_zero(a)


# In[9]:


def increase_matrix_size(current_matrix):
    """Double row count of matrix, retaining old data

    Parameter: current_matrix

    Returns: updated current_matrix with additional rows

    TODO: filter out fully masked rows (broadcast) during copying
    """
    if debug:
        print('Increasing matrix size from', current_matrix.shape)
    current_matrix_row_count, column_count = current_matrix.shape # rows, then columns
    new_matrix_row = ma.MaskedArray(
        data = [-1] * column_count,
        mask = [True] * column_count
    )
    new_matrix = ma.MaskedArray(
        data = np.append(
            current_matrix.data,
            [new_matrix_row] * current_matrix_row_count,
            0 # rows, not columns
        ),
        mask = np.append(
            current_matrix.mask,
            [new_matrix_row.mask] * current_matrix_row_count,
            0
        )
    )
    return new_matrix


# ## Load data

# In[10]:


get_ipython().run_line_magic('run', 'create-blocks.ipynb # data will be in prioritized_blocks')
print('aligning', how_many_paragraphs, 'paragraphs') # confirm


# ## Create witness_node (dataclass)

# In[11]:


@dataclass(unsafe_hash=True)
class WitnessNode:
    """TODO: Write a docstring some day"""
    token_string: str
    witness: str
    witness_offset: int

    def __repr__(self):
        return self.token_string

# Subclass of WitnessNodeEnd so that our constructed end node will be able to identify itself
class WitnessNodeEnd(WitnessNode):
    """TODO: Write a docstring some day"""


# In[12]:


# Store all witness nodes in dictionary of lists
witness_node_lists = defaultdict(list) # one per witness (keys will be sigla)
# create, for each witness:
#   nodes for real tokens
#   END node
# A start node is not created. That is on purpose.
for index, witness_siglum in enumerate(witness_sigla):
    # witness_sigla is a global set when the input data is read
    for witness_token_offset, witness_token in enumerate(witnesses[index]): # list of tokens in a single witness
        witness_node_lists[witness_siglum].append(WitnessNode(witness_token, witness_siglum, witness_token_offset))
    witness_node_lists[witness_siglum].append(WitnessNodeEnd('END', witness_siglum, len(token_array)))


# ## Create candidate vectors from block

# In[13]:


def create_candidate_vectors(block):
    """Returns list of data for block, one numpy array per hyperedge"""
    candidate_vectors = [] # list of individual vectors, not a matrix
    for token_offset_in_block in range(block.token_count):
        # first add the token_offset_in_block to the block start positions
        offset_start_positions = [value + token_offset_in_block for value in block.all_start_positions]
        tokens_to_place = [ # list of all token positions, not just first in each witness
            (
                token_membership_array[token_array_index],
                token_witness_offset_array[token_array_index]
            )
            for token_array_index in offset_start_positions
        ]

        data_for_new_vector = [0] * len(witness_sigla) # initialize to meaningless values
        mask_for_new_vector = [True] * len(witness_sigla) # we'll unmask individual values as needed
        for witness_number, witness_offset in tokens_to_place:
            data_for_new_vector[witness_number] = witness_offset
            mask_for_new_vector[witness_number] = False
        candidate_vectors.append(ma.MaskedArray(data=data_for_new_vector, mask=mask_for_new_vector))

    return candidate_vectors


# ## Place vectors if allowed

# In[14]:


def check_whether_okay_to_place(current_vectors, potential_vector) -> bool:
    """Return True iff we can add row without creating transpositions

    current_vectors: vector space before new addition
    potential_vector: we check whether this can be added

    If subtracting a potential vector from any existing vector would return
    values that diverge in sign, the potential would cross the existing one

    If it's okay to place, we need to call merge_vector() to see whether we
    should merge"""
    subtractionResult = current_vectors - potential_vector
    signs = np.sign(subtractionResult)
    okayToPlace = (signs.min(axis=1) == signs.max(axis=1)).all()
    return True if okayToPlace is ma.masked else okayToPlace


# In[15]:


def merge_vectors(existing_vector: ma.MaskedArray, new_vector: ma.MaskedArray) -> ma.MaskedArray:
    """Combine non-masked values of two vectors, returns one vector

    Sample input:
        v_candidate = ma.MaskedArray(data=[-1, 21, 22, -1], mask=[True, False, False, True])
        v_existing = ma.MaskedArray(data=[-1, -1, 22, 23], mask=[True, True, False, False])

    Sample result:
        masked_array(data=[--, 21, 22, 23], mask=[ True, False, False, False])
        data: non-masked values of two vectors, some of which were already in both
        mask: mask only positions that were masked in both input vectors

    NB:
        does not trap bad data (input vectors that have different non-masked values in same positions)

    """
    v_new = ma.MaskedArray(
        data=np.maximum(
            existing_vector.filled(-1).data,
            new_vector.filled(-1).data),
        mask=(
            existing_vector.mask * new_vector.mask),
        fill_value=-1
    )
    return v_new


# ma1_data = ma1.filled(-1)
# ma2_data = ma2.filled(-1)
# max_values = np.maximum(ma1_data, ma2_data)
# merged_mask = ma1.mask * ma2.mask
# merged_result = ma.MaskedArray(data=max_values, mask=merged_mask, fill_value=-1)
# print(merged_result.data)
# merged_result


# In[16]:


def add_new_vector(input_tuple, new_vector):
    """Add row to matrix and update pointer to next empty row

    Parameters:
        input_tuple : current_matrix, pointer to next empty row
        new_vector : masked array vector to add as new row

    Returns tuple of:
        updated matrix, updated pointer
    """
    current_matrix, pointer = input_tuple
    current_matrix_row_count = current_matrix.shape[0]
    if pointer == current_matrix_row_count: # need more rows now!!!
        current_matrix = increase_matrix_size(current_matrix)
    current_matrix[pointer] = new_vector
    pointer += 1
    return (current_matrix, pointer)


# In[17]:


# Operates on masked arrays, so cannot be compiled with numba
# def create_filter(current_matrix, candidate):
#     #
#     # For each step, handle mask separately from data
#     #
#     # Subtract candidate from matrix (matrix of numbers)
#     #
#     subtracted_data = ma_subtract(current_matrix.data, candidate.data)
#     subtracted_mask = ma_compute_either_masked(current_matrix.mask, candidate.mask)
#     subtraction_result = ma.MaskedArray(data=subtracted_data, mask=subtracted_mask, fill_value=-1)
#     # print(subtraction_result)
#     #
#     # Compute product by row of result of subtraction (masked vector of numbers)
#     #
#     product_data = ma_prod(np.asarray(subtraction_result.filled(1).data), 1)
#     product_mask = np.all(subtraction_result.mask, 1)
#     #
#     # Compute whether row contains zero value (non-masked vector of booleans)
#     #
#     zero_test_data = ma_compare_to_zero(product_data)
#     # print(f"{zero_test_data=}")
#     # print(f"{product_mask=}")
#     zero_test_result = zero_test_data & np.logical_not(product_mask)
#     #     return np_prod(current_matrix - candidate, 1) == 0 # vector of booleans, second argument is axis
#     return zero_test_result # vector of booleans


# In[2]:


@nb.njit(parallel=True)
def create_filter(current_matrix_data, current_matrix_mask, candidate_data, candidate_mask):
    filter = [] # will hold vector of booleans, True if row contains unmasked zero value
    row_count = current_matrix_data.shape[0]
    subtracted_data = current_matrix_data - candidate_data
    subtracted_mask = current_matrix_mask | candidate_mask
    for row_number in nb.prange(row_count):
        current = zip(subtracted_data[row_number], subtracted_mask[row_number])
        non_masked = [item[0] for item in current if not item[1]] # item[1] is True if value is masked
        if 0 in non_masked:
            filter.append(True)
        else:
            filter.append(False)
    return filter
# Would this be better without any numpy functions, e.g., not parallelizing subtraction


# In[3]:


t_current_matrix = ma.MaskedArray(
    data= [[1, 2, 3, 4],
          [-1, -1, -1, -1],
          [1, -1, -1, -1],
          [3,5,6,4]],
    mask=[[False, False, False, False],
         [True, True, True, True],
         [False, True, True, True],
         [False, False, False, False]],
    fill_value = -1
)
t_candidate = ma.MaskedArray(
    data=[1, 2,- 1, -1],
    mask=[False, False, True, True],
    fill_value = -1
)
result = create_filter(t_current_matrix.data, t_current_matrix.mask, t_candidate.data, t_candidate.mask)
print(f"{result=}")

# This is done. Does it work?
#
# Rewritten create_filter() function should return vector of booleans:
# True iff the product of the row is 0
# False if the product of the row is not 0
# False if the entire row is masked


# In[20]:


def add_or_merge_new_vector_into_matrix(input_tuple, candidate):
    """Return max row values (copy) and indices of rows to update in existing matrix

    Parameters:
        input_tuple : (existing matrix, pointer to next empty row in matrix)
        candidate: new vector

    Returns tuple of:
        merged_vector : vector to replace first row to update
        indices : vector of offsets of rows to update (first) or mask (others)

    filter contains vector of booleans, with True for rows in current that are merge candidates
    """

    current_matrix, pointer = input_tuple
    # Use arithmetic instead of comparison where possible
    filter = create_filter(current_matrix.data, current_matrix.mask, candidate.data, candidate.mask)
#    filter = np.prod(current_matrix - candidate, axis=1) == 0 # vector of booleans
#    filter = np.prod(current_matrix[:pointer, :] - candidate, axis=1) == 0
    indices = np.where(filter == True)[0] # row numbers where boolean is True
    if indices.size == 0: # if indices is empty, add new row, update current_matrix and pointer
        current_matrix, pointer = add_new_vector(input_tuple, candidate)
        return (current_matrix, pointer)
    else: # if indices is populated, we merge
#        max_row_values = ma.max(current_matrix[:pointer, :][filter], axis=0) # merger of existing, not yet candidate
        max_row_values = ma.max(current_matrix[filter], axis=0) # merger of existing, not yet candidate
#         if max_row_values.count() == current_matrix.shape[1]: # if max_row_values is fully populated, no need to merge
#             pass
#         else:
        merged_vector = merge_vectors(max_row_values, candidate) # eventual replacement for one of the existing candidates
        new_row = merged_vector
        rows_to_change = indices
        current_matrix[rows_to_change[0]] = new_row # replace first row to replace with merge
        column_count = current_matrix.shape[1] # get column count
        current_matrix[rows_to_change[1:]] = ma.MaskedArray( # mask other rows to replace
            data=[-1] * column_count,
            mask=[True] * column_count
        )
        return (current_matrix, pointer)


# ## Process blocks

# In[21]:


def process_blocks(input_tuple, selected_blocks):
    #current_matrix, pointer = input_tuple
    for index, selected_block in enumerate(selected_blocks):
        # Create vectors for entire block
        new_vectors = create_candidate_vectors(selected_block)
        # Check only first and last, merge is both are okay
        # NB: Checking only first and last to test entire block will break with discontinuous blocks
        #   (should we ever switch to working with discontinuous blocks)
        if input_tuple[1] == 0: # no need to check for transpositions the first time; just place
            for potential_vector in new_vectors:
                input_tuple = add_new_vector(input_tuple, potential_vector)
        else:
            no_transposition =                 check_whether_okay_to_place(input_tuple[0], new_vectors[0]) and                 check_whether_okay_to_place(input_tuple[0], new_vectors[-1])
            if no_transposition: # all-or-nothing
                for potential_vector in new_vectors:
                    input_tuple = add_or_merge_new_vector_into_matrix(input_tuple, potential_vector)
            else:
                pass
    return input_tuple[0] # return updated matrix; TODO: do we need to return pointer, too?


# ## Main

# In[ ]:


# create all vectors (gingerly)
# We exclude blocks with repetition (temporarily?)
selected_blocks = filter(lambda x: len(x.all_start_positions) == x.witness_count, prioritized_blocks)

# set up matrix for vectors
# column count is number of witnesses
# initial row count is equal to length of longest witness
witness_count = len(witness_node_lists)
max_witness_length = max([len(witness_node_lists[w]) for w in witness_node_lists])
alignment_matrix = ma.MaskedArray(
    data = [ma.MaskedArray(
        data = [-1] * witness_count,
        mask = [True] * witness_count
    )] * max_witness_length
)
pointer = 0

get_ipython().run_line_magic('lprun', '-f add_or_merge_new_vector_into_matrix process_blocks((alignment_matrix, pointer), selected_blocks)')
# alignment_matrix = process_blocks((alignment_matrix, pointer), selected_blocks)


# In[ ]:


alignment_matrix


# ## Create variant graph from numpy masked array

# In[ ]:


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


# In[ ]:


# Create networkx variant graph from our (not networkx) alignment graph
# Create variant graph edges for start and end nodes (only)
# We've treated VG as a global and we refer to it in the earlier functions; should we pass it instead?
def augment_or_add_edge_without_conversion(siglum, source_VG_node, target_VG_node):
    if VG.has_edge(source_VG_node, target_VG_node):
        VG[source_VG_node][target_VG_node]["siglum"].append(siglum)
    else:
        VG.add_edge(
            source_VG_node,
            target_VG_node,
            siglum = [siglum]
        )


# In[ ]:


# Use alignment graph node to look up variant graph node
def alignment_node_to_VG_node(alignment_node: WitnessNode) -> VG_node:
    global witness_offset_to_VG_node
    return witness_offset_to_VG_node[alignment_node.witness][alignment_node.witness_offset]


# In[ ]:


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


# In[ ]:


def create_variant_graph_from_vector_space(alignment_matrix):
    # create variant graph and add start and end nodes

    start_node = VG_node("START", {})
    end_node = VG_node("END", {})
    global VG # does this have to be global; we return it at the end, but we also use it in functions above
    VG = nx.DiGraph(start = start_node, end = end_node) # create start and end properties to find terminal nodes
    VG.add_node(start_node)
    VG.add_node(end_node)

    # keep track of which witness nodes (and therefore also hyperedges)
    # have been added to VG
    # does this duplicate information available from the following structure, since
    #   if we assign a value below to replace the None, this doesn't seem to add anything
    #   Can we get rid of this and use witness_offset_to_VG_node for this purpose instead?
    # NB: Includes START and END, which we will later ignore
    from bitarray import bitarray
    global VG_tracking # keep this, but should we return it instead of making it global?
    VG_tracking = {}
    for siglum, witness_node_list in witness_node_lists.items():
        VG_tracking[siglum] = bitarray(len(witness_node_list))
        VG_tracking[siglum].setall(0)

    # map from witness token node to variant graph node (needed to construct edges)
    # values will be added as we create variant graph nodes
    # NB: Includes START and END, which we will later ignore
    global witness_offset_to_VG_node # does this have to be global?
    witness_offset_to_VG_node = {}
    for siglum in witness_node_lists.keys():
        witness_offset_to_VG_node[siglum] = [None] * (len(witness_node_lists[siglum]))

    # Replaces old code in following cell to build variant graph,
    #   this time using vector space as main data source

    # 1. Convert vectors to variant graph nodes

    for row in alignment_matrix:
        if debug:
            print(' '.join(('Processing', str(row))))
        if row.all() is ma.masked: # don't process fully masked rows
            continue
        else:
            # create dictionary of siglum:value for node (variable name: data) and update globals
            data = {}
            for index, value in enumerate(row):
                if value != alignment_matrix.fill_value:
                    siglum = sigla[index]
                    value = int(value)
                    data[siglum] = value # add to eventually data for new VG node
                    VG_tracking[siglum][value] = 1 # update global; do we need to adjust by 1?
            # get token string for node (variable name: token_string)
            siglum, offset = next(iter(data.items()))
            token_string = witness_node_lists[siglum][int(offset)].token_string
            # create and add new VG_node
            new_VG_node = VG_node(token_string, data)
            VG.add_node(new_VG_node)
            for siglum, offset in data.items(): # Eek! Another for loop! How embarrassing!
                witness_offset_to_VG_node[siglum][offset] = new_VG_node # update other global;

    # 2. Traverse variant graph (in arbitrary order), each of which contains witness nodes.
    #    Draw outgoing edges, which point to nodes with next tokens in each witness present
    #      on the variant graph node.
    #    Creating new variant graph nodes for witness tokens not in a hyperedge.
    #    Use queue because we're iterating over a dynamic structure (inventory of variant graph nodes).

    VG_node_queue = queue.Queue()
    #
    # Temporarily skipping START and END, which are the first two nodes
    #
    for node in VG.nodes(): # add all initial VG nodes to queue except START and END
        VG_node_queue.put(node)
        if debug:
            print(f"Adding node to VG_node_queue; length is {VG_node_queue.qsize()=}")
    ignore_start = VG_node_queue.get() # temporarily ignore START node when creating edges
    ignore_end = VG_node_queue.get() # temporarily ignore END node when creating edges
    if debug:
        print(f"Removed START and END; length is {VG_node_queue.qsize()=}")
    while not VG_node_queue.empty():
        if debug:
            print(f"Processing source node from VG_node_queue; length is {VG_node_queue.qsize()=}")
        source_node = VG_node_queue.get()
        targets = set()
        edge_labels = defaultdict(list) # key is target VG node, value is list of sigla
        source_sigla = source_node._sigla # all sigla on source node
        for siglum in source_sigla:
            # target may or may not already exist as node in VG
            source_offset = source_node[siglum]
            target_offset = source_offset + 1
            target = witness_offset_to_VG_node[siglum][target_offset]
            if target: # does the target already exist?:
                targets.add(target)
                edge_labels[target].append(siglum)
            else: # 1) create target, 2) add to targets, 3) add to queue of nodes, and 4 + 5) update globals
                # TODO: Do we need to update globals? We shouldn't need to return to nodes we add here.
                # NB: If real target object is of type WitnessNodeEnd, don't add it to the queue or create an edge
                # print(f"{type(witness_node_lists[siglum][target_offset])=}")
                witness_node_target = witness_node_lists[siglum][target_offset]
                if type(witness_node_target) != WitnessNodeEnd:
                    target_token_string = witness_node_target.token_string
                    new_VG_node = VG_node(target_token_string, {siglum: target_offset}) # 1 create target
                    targets.add(new_VG_node) # 2 add to targets
                    edge_labels[new_VG_node].append(siglum)
                    VG_node_queue.put(new_VG_node) # 3 add to queue of nodes
                    VG_tracking[siglum][target_offset] = 1 # 4 update first global
                    witness_offset_to_VG_node[siglum][target_offset] = new_VG_node # 5. update second global
        for target in targets:
            VG.add_edge(source_node, target, label=",".join(edge_labels[target]))
        #
        # Add edges for start VG nodes
        #
        all_first_data_nodes = defaultdict(list)
        for siglum in witness_sigla:
            key =  witness_offset_to_VG_node[siglum][0]
            all_first_data_nodes[key].append(siglum)
        for key,value in all_first_data_nodes.items():
            VG.add_edge(start_node, key, label=",".join(value))

        #
        # Add edges for end VG node
        #
        all_end_data_nodes = defaultdict(list)
        for siglum in witness_sigla:
            # Figure out why -2 works and clean up as needed
            key = witness_offset_to_VG_node[siglum][-2]
            all_end_data_nodes[key].append(siglum)
        for key,value in all_end_data_nodes.items():
            VG.add_edge(key, end_node, label=",".join(value))

    # we're done! return the result
    return VG


# In[ ]:


# create output
VG = create_variant_graph_from_vector_space(alignment_matrix)


# ## Visualize variant graph

# In[ ]:


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
    if node is None:
        node_text = 'NONE'
    else:
        node_text = node.token_string + " (" + node_id + ")"
    a.node(node_id, label=node_text)

# plot edges
for edge in VG.edges(data=True):
    # edge is a three-item tuple: source, target, dictionary of properties
    a.edge(node_to_id[edge[0]], node_to_id[edge[1]], label=edge[2]["label"])

# print('aligning', how_many_paragraphs, 'paragraphs') # confirm
SVG(a.view())


# In[ ]:


#  This function joins the variant graph in place.
#  This function is a straight port of the Java version of CollateX.

def join(graph):
    processed = set()
    end = graph.graph["end"]
    queue = deque()
    for (_, neighbor) in graph.out_edges(graph.graph["start"]):
        queue.appendleft(neighbor)
    while queue:
        vertex = queue.popleft()
        out_edges = graph.out_edges(vertex)
        if len(out_edges) == 1:
            (_, join_candidate) = next(iter(out_edges))
            can_join = join_candidate != end and len(graph.in_edges(join_candidate)) == 1
            if can_join:
                join_vertex_and_join_candidate(graph, join_candidate, vertex)
                # we have merged join_candidate (tokens originally to the right) into vertex (tokens originally to the left)
                # (for now, join_candidate node and all of its edges are still there, and we need to remove edges before
                #   we can remove node, which we do in a for loop)
                #
                # RESUME HERE: both of the following branches are wrong, the first cryptically (it shows sigla, but not
                #   all sigla) and the second conspicuously
                #
                for (_, neighbor, data) in list(graph.out_edges(join_candidate, data=True)):
                    graph.remove_edge(join_candidate, neighbor)
                    if graph.has_edge(vertex, neighbor):
                        continue # TODO: this looks wrong
#                         graph[source][target]["siglum"].append(data["siglum"])
                    else:
#                         graph.add_edge(vertex, neighbor, siglum=data["siglum"])
                         graph.add_edge(vertex, neighbor,label=data["label"])
                graph.remove_edge(vertex, join_candidate)
                graph.remove_node(join_candidate)
                queue.appendleft(vertex)
                continue
        processed.add(vertex)
        for (_, neighbor) in out_edges:
            # FIXME: Why do we run out of memory in some cases here, if this is not checked?
            if neighbor not in processed:
                queue.appendleft(neighbor)


def join_vertex_and_join_candidate(graph, join_candidate, vertex):
    # Note: since there is no normalized/non normalized content in the graph
    # a space character is added here for non punctuation tokens

    if re.match(r'^\W', join_candidate.token_string):
        vertex.token_string += join_candidate.token_string
    else:
        vertex.token_string += (" " + join_candidate.token_string)
    # join_candidate must have exactly one token (inside a list); left item may have more
#     for siglum, token in join_candidate.tokens.items():
#         vertex.add_token(siglum, token[0])


# In[ ]:


# len(nx.algorithms.cycles.find_cycle(VG))


# In[ ]:


# nx.algorithms.cycles.find_cycle(VG)


# In[ ]:


join(VG)


# In[ ]:


# pp.pprint([edge for edge in VG.edges()])


# In[ ]:


## node id values must be strings for graphviz
a = graphviz.Digraph(format="svg", name="variant_graph_joined")
a.attr(rankdir = "LR")
a.attr(rank = 'same')
a.attr(compound='true')

# plot nodes, building {node: id} for lookup
node_to_id = {}
for index, node in enumerate(VG.nodes()):
    node_id = str(index)
    node_to_id[node] = node_id
    if node is None:
        node_text = 'None'
    else:
        node_text = node.token_string + " (" + node_id + ")"
    a.node(node_id, label=node_text)

# plot edges
for edge in VG.edges(data=True):
    # edge is a three-item tuple: source, target, dictionary of properties
#     label = "(all)" if len(edge[2]["siglum"]) == len(witness_sigla) else ",".join(sorted(edge[2]["siglum"]))
    label = edge[2]["label"]
    a.edge(node_to_id[edge[0]], node_to_id[edge[1]], label=label)

print('aligning', how_many_paragraphs, 'paragraphs') # confirm
SVG(a.view())


# In[ ]:


# Find first non-masked value in each column to avoid traversing entire column
# In our matrix, all non-masked column values for rows that are merge candidates
#   have the same value. Previously we found max(), which had to look at them all
#   and compare, but the first value will necessarily also be the max().

current_matrix = ma.MaskedArray(
    data =[[-1, 2, 3], [4, -1,6], [7,8,-1]],
    mask=[[True, False, False],[False, True, False],[False, False, True]],
    fill_value=-1
)
columns = current_matrix.T # swap rows and columns and process (new) rows
[print(column) for column in columns]
first_non_masked = [column[column.mask == False][0] for column in columns]
first_non_masked


# In[ ]:




