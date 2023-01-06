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
import pprint
from alignment_tree import * # TODO: Fix indirect import of create_token_array()
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

    Input: Tokens of individual unaligned node as list of token lists

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

    Input: linkage object

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

    Input: linkage object

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
        Range of cluster numbers to test (1 < k < n - 1)
        Dictionary of silhouette values by k

    Returns: Void (renders histogram)
    """
    plt.bar(silhouette_range, silhouette_values)
    plt.xlabel('Number of clusters', fontsize = 20)
    plt.ylabel('S(i)', fontsize = 20)
    plt.show()


def group_readings_by_cluster(linkage_object, silhouette):
    """Return mapping of readings to clusters

    Input:
        Linkage object
        Number of clusters (from silhouette)

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


for node in darwin: # Each unaligned zone is its own node
    current_node = node["readings"] # list of lists
    current_linkage_object, current_cophenetic = create_linkage_object(current_node)
    # current_silhouette, current_silhouette_range = compute_silhouette_cutoff(current_linkage_object)
    # readings_by_cluster = group_readings_by_cluster(current_linkage_object, current_silhouette)
    if node["nodeno"] == 1146:
        # print(node["nodeno"], readings_by_cluster, current_silhouette, current_cophenetic)
        print(node["nodeno"], current_cophenetic)
        for witness_number, witness_data in enumerate(current_node):
            print(witness_number, ': ', ' '.join(witness_data))
        print(current_linkage_object)
        render_dendrogram(current_linkage_object)
        merge_stages = {} # alignment trees for merged nodes
        for row_number, row in enumerate(current_linkage_object):
            new_node_number = len(current_node) + row_number
            if row[0] < len(current_node) and row[1] < len(current_node):
                merge_stages[new_node_number] = align_two_readings([current_node[int(row[0])], current_node[int(row[1])]])
            elif row[0] < len(current_node) or row[1] < len(current_node):
                merge_stages[new_node_number] = "Merge singleton into alignment tree"
            else:
                merge_stages[new_node_number] = "Merge two alignment trees"
        for key, value in merge_stages.items():
            print(key, value)