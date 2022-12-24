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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.cluster import AgglomerativeClustering as ag
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import json

with open("unaligned_data.json", "r") as f:
    darwin = json.load(f)

for node in darwin:
    if node["nodeno"] == 125:
        node125 = node["readings"] # list of lists

# corpus = [
#     ['this', 'is', 'the', 'first', 'document', '.'],
#     ['this', 'document', 'is', 'the', 'second', 'document', '.'],
#     ['and', 'this', 'is', 'the', 'third', 'one', '.'],
#     ['is', 'this', 'the', 'first', 'document', '?'],
# ]
def dummy(doc):
    return doc

# vectorizer = CountVectorizer(tokenizer=dummy, preprocessor=dummy)
# X = vectorizer.fit_transform(corpus)
# feature_names = vectorizer.get_feature_names_out()
# feature_matrix = X.toarray()
# print("Feature Names")
# print(feature_names)
# print("Feature Matrix (doc, feature)")
# print(feature_matrix)

darwin_vectorizer = CountVectorizer(tokenizer=dummy, preprocessor=dummy)
darwin_X = darwin_vectorizer.fit_transform((node125))
darwin_feature_matrix = darwin_X.toarray()
print("Darwin Feature Matrix (witness, token)")
print(darwin_feature_matrix)
darwin_feature_matrix_normalized = preprocessing.normalize(darwin_feature_matrix)
print("With normalization")
print(darwin_feature_matrix_normalized)
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

darwin_Z = linkage(darwin_feature_matrix, 'ward')
darwin_Z_normalized = linkage(darwin_feature_matrix_normalized, 'ward')
print("scipy: Agglomerative Clustering")
print(darwin_Z)
print("With normalization")
print(darwin_Z_normalized)
darwin_c, darwin_coph_dists = cophenet(darwin_Z, pdist(darwin_feature_matrix))
darwin_c_normalized, darwin_coph_dists = cophenet(darwin_Z_normalized, pdist(darwin_feature_matrix_normalized))
print("Cophenetic Correlation Coefficient")
print(darwin_c)
print("With normalization")
print(darwin_c_normalized)