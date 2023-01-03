# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
# https://www.cse.uoi.gr/~tsap/teaching/cse012/tutorials/Introduction-SciKit-Learn-Clustering.html
# https://towardsdatascience.com/9-distance-measures-in-data-science-918109d069fa#:~:text=the%20intersection%20similar.-,Use%2DCases,predicted%20segment%20given%20true%20labels.

# example from https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# modified according to https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
# to use list of token lists instead of list of strings
#
# Compute matrix distances according to
# https://www.cse.uoi.gr/~tsap/teaching/cse012/tutorials/Introduction-SciKit-Learn-Clustering.html
#
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
#
# Normalization: https://ryanwingate.com/intro-to-machine-learning/unsupervised/hierarchical-clustering/
# See also:
# https://towardsai.net/p/data-science/scaling-vs-normalizing-data-5c3514887a84
# https://benalexkeen.com/feature-scaling-with-scikit-learn/
# https://www.digitalocean.com/community/tutorials/standardscaler-function-in-python
# https://www.digitalocean.com/community/tutorials/normalize-data-in-python (about the axis parameter)
# https://twintowertech.com/2020/03/22/automatic-clustering-with-silhouette-analysis-on-agglomerative-hierarchical-clustering/

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

with open("../unaligned_data.json", "r") as f:
    darwin = json.load(f)

for node in darwin:
    if node["nodeno"] == 149:
        current_node = node["readings"] # list of lists

def dummy(doc):
    return doc

darwin_vectorizer = CountVectorizer(tokenizer=dummy, preprocessor=dummy)
darwin_X = darwin_vectorizer.fit_transform((current_node))
darwin_feature_matrix = darwin_X.toarray()
# print("Darwin Feature Matrix (witness, token)")
# print(darwin_feature_matrix)
# print(darwin_feature_matrix.shape)
darwin_feature_matrix_normalized = preprocessing.normalize(darwin_feature_matrix)
# print("With normalization")
# print(darwin_feature_matrix_normalized)
# C = metrics.pairwise.cosine_similarity(feature_matrix)
# print('Cosine Similarity')
# print(C)
# D = 1 - C # distances (sort of)

# sklA = ag(n_clusters=2, linkage="ward").fit_predict(C)
# print("sklearn: Agglomerative Clustering")
# print(sklA)

# Z = linkage(feature_matrix, 'ward')
# print("scipy: Agglomerative Clustering")
# print(Z)

# c, coph_dists = cophenet(Z, pdist(feature_matrix))
# print("Cophenetic Correlation Coefficient")
# print(c)

# darwin_Z = linkage(darwin_feature_matrix, 'ward', optimal_ordering=True)
darwin_Z_normalized = linkage(darwin_feature_matrix_normalized, 'ward', optimal_ordering=True)
# print("scipy: Agglomerative Clustering")
# print(darwin_Z)
# print("With normalization")
# print(darwin_Z_normalized)
# darwin_c, darwin_coph_dists = cophenet(darwin_Z, pdist(darwin_feature_matrix))
darwin_c_normalized, darwin_coph_dists = cophenet(darwin_Z_normalized, pdist(darwin_feature_matrix_normalized))
# print("Cophenetic Correlation Coefficient")
# print(darwin_c)
# print("With normalization")
# print(darwin_c_normalized)

# plt.figure(figsize=(25, 10))
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('sample index')
# plt.ylabel('distance')
# dendrogram(
#     darwin_Z_normalized,
#     leaf_rotation=90.,  # rotates the x axis labels
#     leaf_font_size=8.,  # font size for the x axis labels
# )
# plt.show()

k = range(2, 5)
silhouette_scores = {}
silhouette_scores.fromkeys(k)
ag_list = [ag(n_clusters=i) for i in k]
for i, j in enumerate(k):
    silhouette_scores[j] = metrics.silhouette_score(darwin_Z_normalized, ag_list[i].fit_predict(darwin_Z_normalized))
y = list(silhouette_scores.values())
# plt.bar(k, y)
# plt.xlabel('Number of clusters', fontsize = 20)
# plt.ylabel('S(i)', fontsize = 20)
# plt.show()
silhouette = max(silhouette_scores, key=silhouette_scores.get) # key with highest value
print(fcluster(darwin_Z_normalized, silhouette, criterion="maxclust"))
# Silhouette, like elbow, cannot find an endpoint (all singletons or all one cluster)
#   because it evaluates change and the endpoints have no basis for comparison