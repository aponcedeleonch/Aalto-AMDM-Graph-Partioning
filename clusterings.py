from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.spatial.distance import pdist, squareform
from resources import score_function, correct_cluster_labels

def merge_nodes(G, n, k_eig, k, cluster_labels, closest_clusters, logger):
    scores = []
    labels =[cluster_labels]
    #Select nodes
    #Create the new labels
    #Score new labels
    prov_score = score_function(cluster_labels, k, G, logger)
    scores.append(prov_score)
    index = np.argmin(scores)
    return labels[index]

def get_closest_clusters(cluster_centers):
    cluster_distances = pdist(cluster_centers, metric='euclidean')
    cluster_distances = np.triu(squareform(cluster_distances))
    cluster_distances[cluster_distances == 0] = np.Inf
    closest_clusters = np.argwhere(cluster_distances == np.min(cluster_distances))
    return closest_clusters

def cluster_k_means_modified(G, n, k_eig, k, logger):
    cluster_centers, cluster_labels = cluster_k_means(k_eig, k, logger)
    closest_clusters = get_closest_clusters(cluster_centers)
    cluster_labels = correct_cluster_labels(G, cluster_labels)
    best_cluster_labels =  merge_nodes(G, n, k_eig, k, cluster_labels, closest_clusters, logger)
    return best_cluster_labels

def cluster_k_means(k_eig, k, logger):
    logger.info('Using k-means to cluster the vertices')
    kmeans = KMeans(n_clusters=k).fit(k_eig)
    logger.info('K-means finished. Returning the results')
    return kmeans.cluster_centers_, kmeans.labels_

def cluster_agglomerative(k_eig, k, logger, L=None):
    logger.info('Using Agglomerative clustering to cluster the vertices')
    if L is None:
        agglomerative = AgglomerativeClustering(n_clusters=k).fit(k_eig)
    else:
        agglomerative = AgglomerativeClustering(n_clusters=k,
                                                connectivity=L).fit(k_eig)
    logger.info('Agglomerative clustering finished. Returning the results')
    return agglomerative.labels_

def cluster_gmm(k_eig, k, logger):
    logger.info('Using GMM to cluster the vertices')
    gmm = GaussianMixture(n_components=k).fit(k_eig)
    labels = gmm.predict(k_eig)
    logger.info('GMM finished. Returning the results')
    return labels
