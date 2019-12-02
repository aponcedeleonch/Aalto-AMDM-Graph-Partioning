from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import itertools
from scipy.spatial.distance import pdist, cdist, euclidean, squareform
from resources import score_function, correct_cluster_labels

def get_nodes_to_merge(k_eig, cluster_centers,cluster_labels,closest_clusters, n,k ):
    nodes = []
    for i in range (2):
        #Select nodes
        cluster_idx = [idx for idx, val in enumerate(cluster_labels) if val == closest_clusters[0,i]]
        cluster_arr = np.zeros((len(cluster_idx), k))
        #Build arrays of clusters
        for index, value in enumerate(cluster_idx):
            cluster_arr[index,] = k_eig[value,]
        if (i == 0):
            XB = cluster_centers[closest_clusters[0,1]]
        else:
            XB = cluster_centers[closest_clusters[0,0]]
        #Pre allocate
        distance_0 = np.zeros(len(cluster_idx))
        for m in range(len(cluster_idx)):
            distance_0[m] = euclidean(cluster_arr[m], XB)
        relative_index = np.argpartition(distance_0, n)
        relative_index = relative_index[:n]
        nodes_to_merge = np.zeros(len(relative_index))
        for a,b in enumerate(relative_index):
            nodes_to_merge[a] = cluster_idx[b]
        nodes.append(nodes_to_merge)
    return nodes

def merge_nodes(G, n, k_eig, k, cluster_labels, cluster_centers, closest_clusters, logger):
    scores = []
    labels =[]
    logger.info('Choosing nodes to merge')
    nodes_to_merge = get_nodes_to_merge(k_eig, cluster_centers,cluster_labels,closest_clusters, n,k)
    nodes_to_merge = list(itertools.chain.from_iterable(nodes_to_merge))
    combinations = [i for i in itertools.product([closest_clusters[0,0], closest_clusters[0,1]], repeat=2*n)]
    n_labels = len(combinations)
    logger.info('Grading %d combinations'%(n_labels))
    for i in range(n_labels):
        labels.append(cluster_labels)
        for j in range(2*n):
            node = int(nodes_to_merge[j])
            labels[i][node] = combinations[i][j]
        labels[i] = correct_cluster_labels(G, labels[i])
        if (i % int(n_labels/10) == 0):
                logger.info('Grading combination %d.' % (i))
        prov_score = score_function(labels[i], k, G, logger)
        scores.append(prov_score)
    #Score new labels
    index = np.argmin(scores)
    logger.info('Index of best combination %d.' % (index))
    logger.info(combinations[index])
    return labels[index]

def get_closest_clusters(cluster_centers):
    cluster_distances = pdist(cluster_centers, metric='euclidean')
    cluster_distances = np.triu(squareform(cluster_distances))
    cluster_distances[cluster_distances == 0] = np.Inf
    closest_clusters = np.argwhere(cluster_distances == np.min(cluster_distances))
    return closest_clusters

def cluster_k_means_modified(G, n, k_eig, k, logger):
    cluster_centers, cluster_labels = cluster_k_means(k_eig, k, logger)
    logger.info('Getting closest clusters')
    closest_clusters = get_closest_clusters(cluster_centers)
    logger.info('Closest clusters')
    logger.info(closest_clusters)
    #cluster_labels = correct_cluster_labels(G, cluster_labels)
    best_cluster_labels =  merge_nodes(G, n, k_eig, k, cluster_labels, cluster_centers, closest_clusters, logger)
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
        logger.info('Using L as connectivity matrix')
        agglomerative = AgglomerativeClustering(n_clusters=k,
                                                connectivity=L,
                                                linkage='average').fit(k_eig)
    logger.info('Agglomerative clustering finished. Returning the results')
    return agglomerative.labels_

def cluster_gmm(k_eig, k, logger):
    logger.info('Using GMM to cluster the vertices')
    gmm = GaussianMixture(n_components=k).fit(k_eig)
    labels = gmm.predict(k_eig)
    logger.info('GMM finished. Returning the results')
    return labels
