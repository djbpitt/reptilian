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
pp = pprint.PrettyPrinter(2)

with open("unaligned_data.json", "r") as f:
    darwin = json.load(f)

def dummy(doc):
    return doc


def create_linkage_object(input_node):
    darwin_vectorizer = CountVectorizer(tokenizer=dummy, preprocessor=dummy, token_pattern=None)
    # set token_pattern to None to suppress unneeded warnings
    darwin_X = darwin_vectorizer.fit_transform(input_node)
    darwin_feature_matrix = darwin_X.toarray()
    darwin_feature_matrix_normalized = preprocessing.normalize(darwin_feature_matrix)
    darwin_Z_normalized = linkage(darwin_feature_matrix_normalized, 'ward', optimal_ordering=True)
    darwin_c_normalized, darwin_coph_dists = cophenet(darwin_Z_normalized, pdist(darwin_feature_matrix_normalized))
    return darwin_Z_normalized, darwin_c_normalized # linkage object, cophenetic correlation coefficient


def render_dendrogram(current_linkage_object):
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        current_linkage_object,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()


def compute_silhouette_cutoff(current_linkage_object) -> int:
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
    plt.bar(silhouette_range, silhouette_values)
    plt.xlabel('Number of clusters', fontsize = 20)
    plt.ylabel('S(i)', fontsize = 20)
    plt.show()


def group_readings_by_cluster(linkage_object, silhouette):
    groupings = fcluster(linkage_object, silhouette, criterion="maxclust")
    return groupings


for node in darwin:
    if node["nodeno"] < 150:
        current_node = node["readings"] # list of lists
        current_linkage_object, current_cophenetic = create_linkage_object(current_node)
        current_silhouette, current_silhouette_range = compute_silhouette_cutoff(current_linkage_object)
        readings_by_cluster = group_readings_by_cluster(current_linkage_object, current_silhouette)
        if current_silhouette > 2:
            print(node["nodeno"], readings_by_cluster, current_silhouette, current_cophenetic)
            # render_dendrogram(current_linkage_object)
            # render_silhouette_profile(range(2, 5), current_silhouette_range)


