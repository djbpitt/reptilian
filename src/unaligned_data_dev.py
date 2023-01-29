from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.cluster import AgglomerativeClustering as ag
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, ward
import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import json
from collections import deque
import pprint
from typing import Callable
from alignment_tree import * # TODO: Fix indirect import of create_token_array(), find_longest_sequences()
pp = pprint.PrettyPrinter(2)

# ===
# Import data (non-block zones)
# ===
with open("unaligned_data.json", "r") as f:
    darwin = json.load(f)

def dummy(doc):
    """Needed by CountVectorizer()"""
    return doc


def create_linkage_object(input_node):
    """Prepare witness data for clustering

    Input:
        input_node : Tokens of individual unaligned node as list of token lists

    Returns:
        Normalized distance matrix of token counts per witness (Ward linkage method)
        Cophenetic correlation coefficient (for verification)
    """
    darwin_vectorizer = CountVectorizer(tokenizer=dummy, preprocessor=dummy, token_pattern=None)
    # set token_pattern to None to suppress unneeded warnings
    darwin_X = darwin_vectorizer.fit_transform(input_node)
    darwin_feature_matrix = darwin_X.toarray()
    darwin_feature_matrix_normalized = preprocessing.normalize(darwin_feature_matrix)
    darwin_Z_normalized = linkage(darwin_feature_matrix_normalized, 'ward', optimal_ordering=True)
    darwin_c_normalized, darwin_coph_dists = cophenet(darwin_Z_normalized, pdist(darwin_feature_matrix_normalized))
    return darwin_Z_normalized, darwin_c_normalized # linkage object, cophenetic correlation coefficient


def render_dendrogram(current_linkage_object):
    """Render dendrogram of single unaligned node

    Input:
        current_linkage_object : linkage object

    Returns: Void (renders image with matplotlib)
    """
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        current_linkage_object,
        # leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=20.,  # font size for the x axis labels
    )
    plt.show()


def compute_silhouette_cutoff(current_linkage_object) -> int:
    """Apply silhouette method to find optimum number of clusters

    Input:
        current_linkage_object : linkage object

    Returns:
        Optimal number of clusters (key with highest value)
        All values (for rendering)

    Note:
        1 < k < n - 1 (cannot find optimal value of 1 or n)
    """
    k = range(2, 5)
    silhouette_scores = {}
    silhouette_scores.fromkeys(k)
    ag_list = [ag(n_clusters=i) for i in k]
    for i, j in enumerate(k):
        silhouette_scores[j] = metrics.silhouette_score(current_linkage_object, ag_list[i].fit_predict(current_linkage_object))
    y = list(silhouette_scores.values())
    silhouette = max(silhouette_scores, key=silhouette_scores.get) # key with highest value
    return silhouette, y


def render_silhouette_profile(silhouette_range, silhouette_values):
    """Render histogram of silhouette values

    Input:
        silhouette_range : Range of cluster numbers to test (1 < k < n - 1)
        silhouette_values : Dictionary of silhouette values by k

    Returns: Void (renders histogram)
    """
    plt.bar(silhouette_range, silhouette_values)
    plt.xlabel('Number of clusters', fontsize = 20)
    plt.ylabel('S(i)', fontsize = 20)
    plt.show()


def group_readings_by_cluster(linkage_object, silhouette):
    """Return mapping of readings to clusters

    Input:
        linkage_object : Linkage object
        silhouette : Number of clusters (from silhouette)

    Returns: list of cluster numbers, one per reading
    """
    groupings = fcluster(linkage_object, silhouette, criterion="maxclust")
    return groupings


def align_two_readings(readings:list):
    """Align two readings (token lists) by recursive longest common contiguous subsequence

    Input:
        readings: list of two token lists

    Returns: Alignment tree (not variant graph)
    """
    # TODO: This is the same method as in the first pass in reptilian.py, so fold into main code base
    token_array, token_membership_array, token_witness_offset_array, token_ranges = create_token_array(readings)
    # print(f"{token_array=}")
    # print(f"{token_membership_array=}")
    # print(f"{token_witness_offset_array=}")
    # print(f"{token_ranges=}")

    # ###
    # Initialize alignment tree and add root
    # nodes_to_process is queue of nodes to check for expansion
    # (deque for performance reasons; we use only FIFO, so regular queue)
    # ###
    alignment_tree = create_tree()
    alignment_tree.add_node(0, type="potential", token_ranges=token_ranges)
    nodes_to_process = deque([0])

    # ###
    # Expand tree, starting at root
    # ###
    counter = 0
    while nodes_to_process:
        # print('Iteration #', counter)
        counter += 1
        # print("Head of queue: ", alignment_tree.nodes[nodes_to_process[0]]['token_ranges'])
        if counter == 1:  # special handling for root node
            expand_node(alignment_tree,
                        nodes_to_process,
                        token_array,
                        token_membership_array,
                        len(token_ranges))
            continue
        # All nodes except root
        local_token_array = []
        local_token_membership_array = []
        for index, token_range in enumerate(alignment_tree.nodes[nodes_to_process[0]]['token_ranges']):
            local_token_array.extend(token_array[token_range[0]: token_range[1]])
            local_token_membership_array.extend(token_membership_array[token_range[0]: token_range[1]])
            if index < len(token_ranges) - 1:
                local_token_array.append(' #' + str(index + 1) + ' ')
                local_token_membership_array.append(' #' + str(index + 1) + ' ')
        # print("Local token array: ", local_token_array)
        # print("Local token membership array: ", local_token_membership_array)
        expand_node(alignment_tree,
                    nodes_to_process,
                    local_token_array,
                    local_token_membership_array,
                    len(token_ranges))
    return alignment_tree


def check_for_block_contains_witness_and_repetition(_suffix_array: SuffixArray, _token_membership_array: list, _lcp_interval: LcpIntervalCandidate) -> bool:
    """Write a docstring someday

    The first witness (0) is the singleton
    Filter out blocks that have no relation with witness we are adding
    A block should know which witnesses it’s in

    Number of prefixes >= total number of witnesses
    Accumulate set of witness sigla for prefixes
    if:
        no witness occurs more than once, return True to keep this block
    else:
        return False
    """
    # print(f"{_lcp_interval.lcp_start_offset=}")
    # print(f"{_lcp_interval.lcp_end_offset=}") # inclusive
    # print(range(_lcp_interval.lcp_start_offset, _lcp_interval.lcp_end_offset + 1))
    _witnesses_found = []
    for _lcp_interval_item_offset in range(_lcp_interval.lcp_start_offset, _lcp_interval.lcp_end_offset + 1):
        _token_position = _suffix_array.SA[_lcp_interval_item_offset]  # point from prefix to suffix array position
        _witness_siglum = _token_membership_array[
            _token_position]  # point from token array position to witness identifier
        if _witness_siglum in _witnesses_found:
            return False
        else:
            _witnesses_found.append(_witness_siglum)
    if 0 in _witnesses_found:
        return True
    else:
        return False


def create_blocks_for_witness_and_alignment_tree(_suffix_array: SuffixArray, _token_membership_array: list):
    """Write a docstring someday

    The singleton witness is witness 0, identifiable from the token_membership_array
    Other witnesses (two or more) follow

    Look at changes in length of LCP array
    Initial value is 0 or -1 because it's a comparison with previous, and first has no previous
    Next value is number of tokens shared with previous
    Exact length doesn't matter, but if it changes, new pattern:
        If it stays the same, take note but do nothing yet; it means that the pattern repeats
        No change for a while, then goes to 0:
            Number of repetitions plus 1, e.g., 5 5 5 0 = 4 instances of 5
            Once it changes to 0, we've seen complete pattern
        Changer to smaller means hidden, deeper block
        Changes to longer means ???
    """
    _accumulator = []  # lcp positions (not values) since most recent 0
    _frequent_sequences = []  # lcp intervals to be considered for mfs
    _lcp_array = _suffix_array._LCP_values
    #
    # lcp value
    # if == 0 it's a new interval, so:
    #   1. if there is already an accumulation, commit (process) it
    #      "committing the buffer" means checking for repetition and depth
    #          if it passes check: store in mfs list
    #          otherwise throw it away
    #   2. clear buffer (accumulator) and begin accumulating new buffer with the new offset with 0 value
    # otherwise it isn't zero, so there must be a buffer in place, so add to it (for now)
    for _offset, _value in enumerate(_lcp_array):
        if not _accumulator and _value == 0:  # if accumulator is empty and new value is 0, do nothing
            continue
        elif not _accumulator:  # accumulator is empty and new value is non-zero, so begin new accumulator
            _accumulator.append(LcpIntervalCandidate(lcp_start_offset=_offset - 1, lcp_interval_token_count=_value))
        elif _value > _accumulator[-1].lcp_interval_token_count:  # new interval, so add to accumulator and continue
            _accumulator.append(LcpIntervalCandidate(lcp_start_offset=_offset - 1, lcp_interval_token_count=_value))
        elif _value == _accumulator[-1].lcp_interval_token_count:  # same block as before, so do nothing
            continue
        else:  # new value is less than top of accumulator, so pop everything that is higher
            # Positions in lcp array and suffix array coincide:
            #   The lcp array value is the length of the sequence
            #   The suffix array value is the start position of the sequence
            # Assume accumulator values (offsets into lcp array) point to [3, 6] and new value is 4, so:
            #   First: Pop pointer to 6 (length value in lcp array), store in frequent_sequences
            #   Second: Push new pointer to same position in lcp array, but change value in lcp array to 4
            while _accumulator and _accumulator[-1].lcp_interval_token_count > _value:
                # Create pointer to last closed block that is not filtered (like frequent_sequences)
                _newly_closed_block = _accumulator.pop()
                _newly_closed_block.lcp_end_offset = _offset - 1
                if check_for_block_contains_witness_and_repetition(_suffix_array, _token_membership_array, _newly_closed_block):
                    _frequent_sequences.append(
                        [_newly_closed_block.lcp_start_offset, _newly_closed_block.lcp_end_offset,
                         _newly_closed_block.lcp_interval_token_count])
            # There are three options:
            #   1. there is content in the accumulator and latest value is not 0
            #   2. accumulator is empty and latest value is 0
            #   3. accumulator is empty and latest value is not 0
            # (the fourth logical combination, content in the accumulator and 0 value, cannot occur
            #     because a 0 value will empty the accumulator)
            if _value > 0 and (not _accumulator or _accumulator[-1].lcp_interval_token_count != _value):
                _accumulator.append(LcpIntervalCandidate(lcp_start_offset=_newly_closed_block.lcp_start_offset,
                                                         lcp_interval_token_count=_value))
    # End of lcp array; run through any residual accumulator values
    while _accumulator:
        _newly_closed_block = _accumulator.pop()
        _newly_closed_block.lcp_end_offset = len(_lcp_array) - 1
        if check_for_block_contains_witness_and_repetition(_suffix_array, _token_membership_array, _newly_closed_block):
            _frequent_sequences.append([_newly_closed_block.lcp_start_offset, len(_lcp_array) - 1,
                                        _newly_closed_block.lcp_interval_token_count])
    return _frequent_sequences


def get_tokens_for_block(_block: tuple, _suffix_array: SuffixArray, _ta: list):
    """Return tokens for block

    NB: Blocks are not necessarily full-depth or non-repeating here (they are in first-pass version)

    Parameters:
        _block: tuple (see below)
        _sa : suffix array (provides LCP array)
        _ta : full token array of all witnesses in block

    _block is a tuple of:
      length
      value: start positions in each witness (list of ints)
          To examine tokens we need the start position for just the first witness plus the length

    Returns:
        list of tokens

    LCP start and end are 1-based offset into SA array (end position is inclusive)
    """
    _sa = _suffix_array.SA
    _lcp = _suffix_array._LCP_values
    _length, _starts = _block # unpack tuple
    # print(_suffix_array)
    # print(f"{_sa=}")
    # print(f"{_lcp=}")
    # print(f"{_block=}")
    # print(f"{_token_start_offset=}")
    # print(f"{_block[2]=}")
    _start = _starts[0]
    # print(f"{_token_array_offsets=}")
    # print(f"{_token_array_offsets=}")
    # print(f"{_start=}")
    # print(f"{_end=}")
    print(_ta[_start: _start + _length])
    # print(" ".join(_ta[125: 125 + _block[2]]))


def create_alignment_tree_to_token_mapping(_global_token_array_length: int, _existing_alignment_tree,
                                           _token_range_mapping):
    # Existing alignment tree is already sorted
    # Array where each position is a token in the token array, and store there the node to which the token belongs
    # Create mapping between nodes in existing alignment tree and tokens
    # Create list of same length as global (sic) token array,
    #   traverse nodes (except root) in existing alignment tree in order,
    #   determine which tokens they contain,
    #   place node identifier into list for that token.
    # TODO: Wasted space in list for global token array when we're working only with local data
    _nodes_in_existing_alignment_tree = [None] * _global_token_array_length
    for _node in _existing_alignment_tree.nodes(data=True):
        if _node[0] > 0:
            # print(_node[0], _node[1]["token_ranges"])
            for _token_range in _node[1]["token_ranges"]:
                _nodes_in_existing_alignment_tree[_token_range[0]: _token_range[1]] = [_node[0]] * (
                            _token_range[1] - _token_range[0])
    # print(len(_nodes_in_existing_alignment_tree), _nodes_in_existing_alignment_tree)
    return lambda x: _nodes_in_existing_alignment_tree[_token_range_mapping[x]]


def add_reading_to_alignment_tree(_readings:list, _map_local_token_to_alignment_tree):
    """Fold new reading into existing alignment tree

    Input:
        _readings: list of three or more token lists
        _map_local_token_to_alignment_tree: function

    Returns: Alignment tree (not variant graph)
    """
    # TODO: This is the same method as in the first pass in reptilian.py, so fold into main code base
    _token_array, _token_membership_array, _token_witness_offset_array, _token_ranges = create_token_array(_readings)

    new_alignment_tree = create_tree()
    new_alignment_tree.add_node(0, type="potential", token_ranges=_token_ranges)

    # ###
    # Initialize alignment tree and add root
    # nodes_to_process is queue of nodes to check for expansion
    # (deque for performance reasons; we use only FIFO, so regular queue)
    # ###
    new_alignment_tree = create_tree()
    new_alignment_tree.add_node(0, type="potential", token_ranges=_token_ranges)
    _sa = create_suffix_array(_token_array)

    # print(_sa)
    _frequent_sequences = create_blocks_for_witness_and_alignment_tree(_sa, _token_membership_array)
    # print(_frequent_sequences[0]) # List of lists; block is LCP interval start offset, end offset, and token count
    # print(f"{len(_frequent_sequences)=}")
    # longest sequences is a dictionary, where:
    #   key : end position of longest witness (used only to get rid of shorter, embedded sequences)
    #   value: tuple of length (int) and start positions in each witness (list of ints)
    #       To examine tokens we need the start position for just the first witness plus the length
    _longest_sequences = find_longest_sequences(_frequent_sequences, _sa)
    # Sort by order of new witness, breaking ties with total tokens that would be aligned
    # We favor token count over depth because depth has already been encoded in the existing
    #   alignment tree into which we're merging
    #   We don't replicate the extra step in Java used to break ties
    blocks = _longest_sequences.values()
    _sorted_blocks_by_singleton = sort_blocks_by_singleton_order(blocks)
    # for _ss in _sorted_sequences:
    #     print(_ss)
    #     get_tokens_for_block(_ss, _sa, _token_array)
    # Sort blocks (_longest sequences) in alignment_tree order
    # print("***_longest_sequences***")
    # pp.pprint(_longest_sequences)
    # The second value in the block is a position in the token array occupied by a token from one member of the
    #   existing alignment tree. Use that to index into those nodes and it will return a node number.
    # Mind the local to global token position mapping.
    # Order by position in the alignment tree, if equal, order by position in the singleton,
    #                                                       if equal prefer larger block
    _sorted_blocks_by_alignment_tree = sort_blocks_by_alignment_tree_order(blocks, _map_local_token_to_alignment_tree)
    print("***_sorted_blocks_by_alignment_tree***")
    pp.pprint(_sorted_blocks_by_alignment_tree)
    print("***_sorted_blocks_by_singleton***")
    pp.pprint(_sorted_blocks_by_singleton)

    # ###
    # RESUME HERE
    ###
    #
    # Break the sort routines into separate functions and test them.
    #
    # We should revisit and scrutinize the beam search, as well.
    ###
    return

    # ###
    # Expand tree, starting at root
    # ###
    counter = 0
    while nodes_to_process:
        # print('Iteration #', counter)
        counter += 1
        # print("Head of queue: ", alignment_tree.nodes[nodes_to_process[0]]['token_ranges'])
        if counter == 1:  # special handling for root node
            expand_node(new_alignment_tree,
                        nodes_to_process,
                        token_array,
                        token_membership_array,
                        len(token_ranges))
            continue
        # All nodes except root
        local_token_array = []
        local_token_membership_array = []
        for index, token_range in enumerate(new_alignment_tree.nodes[nodes_to_process[0]]['token_ranges']):
            local_token_array.extend(token_array[token_range[0]: token_range[1]])
            local_token_membership_array.extend(token_membership_array[token_range[0]: token_range[1]])
            if index < len(token_ranges) - 1:
                local_token_array.append(' #' + str(index + 1) + ' ')
                local_token_membership_array.append(' #' + str(index + 1) + ' ')
        # print("Local token array: ", local_token_array)
        # print("Local token membership array: ", local_token_membership_array)
        expand_node(new_alignment_tree,
                    nodes_to_process,
                    local_token_array,
                    local_token_membership_array,
                    len(token_ranges))
    return new_alignment_tree


def sort_blocks_by_singleton_order(blocks):
    return sorted(blocks, key=lambda x: (x[1][0], -x[0]))


def sort_blocks_by_alignment_tree_order(blocks, _map_local_token_to_alignment_tree:Callable[[int], int]):
    """Returns function to sort alignment tree nodes

    Parameters:
        blocks : list of two-item tuples (token length, [start offsets for all witnesses])
        _map_local_token_to_alignment_tree : anonymous function with
            one argument: token position in local token array as int
            returns: node identifier in alignment tree as int

    Returns:
        List of blocks ordered by three factors:
            Order in alignment tree
            Singleton order
            Length of block (we don't care about depth here because we've already accounted for it)
    # TODO: Replace inner parameters to second argument with type aliases
    """
    _sorted_blocks_by_alignment_tree = sorted(blocks,
                                              key=lambda x: (
                                                  _map_local_token_to_alignment_tree(x[1][1]),
                                                  x[1][0],
                                                  -x[0]))
    return _sorted_blocks_by_alignment_tree


for node in darwin: # Each unaligned zone is its own node
    readings = node["readings"] # list of lists
    current_linkage_object, current_cophenetic = create_linkage_object(readings)
    # current_silhouette, current_silhouette_range = compute_silhouette_cutoff(current_linkage_object)
    # readings_by_cluster = group_readings_by_cluster(current_linkage_object, current_silhouette)
    if node["nodeno"] == 1146:
        # print(node["nodeno"], readings_by_cluster, current_silhouette, current_cophenetic)
        # print(node["nodeno"], current_cophenetic)
        global_token_array, global_token_membership_array, global_token_witness_offset_array, global_token_ranges = create_token_array(readings)
        # print(f"{global_token_array=}")
        # print(f"{global_token_membership_array=}")
        # print(f"{global_token_witness_offset_array=}")
        print(f"{global_token_ranges=}")
        # for witness_number, witness_data in enumerate(current_node):
        #     print(witness_number, ': ', ' '.join(witness_data))
        print(current_linkage_object)
        render_dendrogram(current_linkage_object)
        merge_stages = {} # alignment trees for merged nodes
        for row_number, row in enumerate(current_linkage_object):
            # In a linkage object the columns are:
            #   0 : (super)witness id
            #   1 : (super)witness id
            #   2 : distance
            #   3 : number of original witnesses in resulting cluster
            # Witness id is:
            #   Real witness if less than total witness count
            #   Constructed superwitness if greater than or equal to witness count
            #   To get from witness id to original witnesses:
            #       Witness id less than total witness count is original witness
            #       Otherwise, witness id - total witness count - 1 = row in linkage object
            #           Apply recursively and collect all values less than witness count
            new_node_number = len(readings) + row_number
            row_0_witness_id = int(row[0])
            row_1_witness_id = int(row[1])
            witness_ids = (row_0_witness_id, row_1_witness_id)
            if row_0_witness_id < len(readings) and row_1_witness_id < len(readings):
                # print("Merging two single readings:", row_0_witness_id, 'and', row_1_witness_id)
                interim_alignment_tree = align_two_readings([readings[row_0_witness_id], readings[row_1_witness_id]])
                # print("Ranges for alignment tree:", interim_alignment_tree.nodes[0]["token_ranges"])
                # print("Corresponding global ranges:", (global_token_ranges[row_0_witness_id],
                # global_token_ranges[row_1_witness_id]))
                adjustments_for_witnesses = [g[0] - l[0] for l, g in zip(interim_alignment_tree.nodes[0]["token_ranges"], (global_token_ranges[row_0_witness_id], global_token_ranges[row_1_witness_id]))]
                # print(f"{adjustments_for_witnesses=}")
                for node_no in interim_alignment_tree.nodes:
                    for index, token_range in enumerate(interim_alignment_tree.nodes[node_no]["token_ranges"]):
                        # print("Local token range:", token_range) # local range
                        adjusted_range = tuple(item + adjustments_for_witnesses[index] for item in token_range)
                        # print("Adjusted range:", adjusted_range)
                        # print(global_token_array[adjusted_range[0]: adjusted_range[1]])
                        interim_alignment_tree.nodes[node_no]["token_ranges"][index] = adjusted_range
                merge_stages[new_node_number] = interim_alignment_tree
                # print(interim_alignment_tree.nodes(data=True))
                # print(f"{merge_stages[new_node_number].nodes=}")
                # for node in merge_stages[new_node_number].nodes:
                #     print(merge_stages[new_node_number].nodes[node])
                # print(f"{merge_stages[new_node_number].edges=}")
            elif row_0_witness_id < len(readings) or row_1_witness_id < len(readings):
                # For merged node the tokens are the ranges in the root, which is the nodes[0] property
                print("Merging singleton into alignment tree:", row_0_witness_id, 'and', row_1_witness_id)
                local_readings = []
                # a linear list of integers that maps a position from the local token array to a position
                # in the global token array
                token_range_mapping = []
                for witness_id in sorted([row_0_witness_id, row_1_witness_id]):
                    if witness_id < len(readings): # singleton
                        global_tokens_singleton = readings[witness_id]
                        global_token_range_singleton = global_token_ranges[witness_id]
                        # print("The tokens of the witness to be aligned in the original array is: "+
                        # str(global_tokens_singleton))
                        # print("The token ranges of the witness to be aligned is: "+str(global_token_range_singleton))
                        local_readings.extend([global_tokens_singleton])
                        token_range_mapping.extend(range(global_token_range_singleton[0], global_token_range_singleton[1]))
                    else: # alignment tree
                        for token_range in merge_stages[witness_id].nodes[0]["token_ranges"]:
                            local_readings.extend([global_token_array[token_range[0]: token_range[1]]])
                            if token_range_mapping:
                                token_range_mapping.append(None)  # splitter token
                            token_range_mapping.extend(range(token_range[0], token_range[1]))
                        existing_alignment_tree = merge_stages[witness_id]
                # print("Tokens before we start merging...")
                # print(f"{tokens=}")
                # print("token_range_mapping list looks like this: " + str(token_range_mapping))
                map_local_token_to_alignment_tree = create_alignment_tree_to_token_mapping(len(global_token_array),
                                                                        existing_alignment_tree, token_range_mapping)
                add_reading_to_alignment_tree(local_readings, map_local_token_to_alignment_tree)
                merge_stages[new_node_number] = "Merge singleton into alignment tree"
            else:
                merge_stages[new_node_number] = "Merge two alignment trees"
        print("Merge stages:")
        for key, value in merge_stages.items():
            print(key, value)
        pp.pprint(merge_stages[7])
