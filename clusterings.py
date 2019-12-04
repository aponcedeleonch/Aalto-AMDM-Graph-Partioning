from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import itertools
from scipy.spatial.distance import pdist, cdist, euclidean, squareform
from resources import score_function, correct_cluster_labels


def get_nodes_to_merge(k_eig, cluster_centers, cluster_labels, closest_clusters, n, k):
    nodes = []
    for i in range(2):
        # Select nodes
        cluster_idx = [idx for idx, val in enumerate(cluster_labels) if val == closest_clusters[0, i]]
        k_given = k_eig.shape[1]
        cluster_arr = np.zeros((len(cluster_idx), k_given))
        # Build arrays of clusters
        for index, value in enumerate(cluster_idx):
            cluster_arr[index, ] = k_eig[value, ]
        if (i == 0):
            XB = cluster_centers[closest_clusters[0, 1]]
        else:
            XB = cluster_centers[closest_clusters[0, 0]]
        # Pre allocate
        distance_0 = np.zeros(len(cluster_idx))
        for m in range(len(cluster_idx)):
            distance_0[m] = euclidean(cluster_arr[m], XB)
        relative_index = np.argpartition(distance_0, n)
        relative_index = relative_index[:n]
        nodes_to_merge = np.zeros(len(relative_index))
        for a, b in enumerate(relative_index):
            nodes_to_merge[a] = cluster_idx[b]
        nodes.append(nodes_to_merge)
    return nodes


def merge_nodes(G, n, k_eig, k, cluster_labels, cluster_centers, closest_clusters, logger):
    scores = []
    labels = []
    logger.info('Choosing nodes to merge')
    nodes_to_merge = get_nodes_to_merge(k_eig, cluster_centers, cluster_labels, closest_clusters, n, k)
    nodes_to_merge = list(itertools.chain.from_iterable(nodes_to_merge))
    combinations = [i for i in itertools.product([closest_clusters[0, 0], closest_clusters[0, 1]], repeat=2*n)]
    n_labels = len(combinations)
    logger.info('To grade %d combinations' % (n_labels))
    for i in range(n_labels):
        labels.append(cluster_labels)
        for j in range(2*n):
            node = int(nodes_to_merge[j])
            labels[i][node] = combinations[i][j]
        labels[i] = correct_cluster_labels(G, labels[i])
        if (i % int(n_labels/4) == 0):
            logger.info('Grading combination %d.' % (i))
        prov_score = score_function(labels[i], k, G, logger)
        scores.append(prov_score)
    # Score new labels
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
    # cluster_labels = correct_cluster_labels(G, cluster_labels)
    best_cluster_labels = merge_nodes(G, n, k_eig, k, cluster_labels, cluster_centers, closest_clusters, logger)
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


def merge_clusters(G, final_k, node_cluster, num_clusters, merge_need, logger):
    logger.info('Number of merges to go: %d' % (merge_need))

    if merge_need == 0:
        logger.info('Returning final clustering')
        return node_cluster

    logger.info('Getting the number of cutting edges between all the clusters')
    cutting_edges = np.zeros((num_clusters, num_clusters))
    cluster_sizes = {}
    for node, cluster in node_cluster.items():
        # Count the sizes of the clusters
        if cluster in cluster_sizes:
            cluster_sizes[cluster] += 1
        else:
            cluster_sizes[cluster] = 1
        for neigh in G[str(node)]:
            cluster_of_neigh = int(node_cluster[int(neigh)])
            cluster_of_node = int(cluster)
            if cluster_of_neigh != cluster_of_node:
                cutting_edges[cluster_of_node, cluster_of_neigh] += 1

    logger.debug('Symmetrical matrix. Getting only upper part')
    cutting_edges = np.triu(cutting_edges)
    max_cutting_edges = cutting_edges.max()
    max_cutting_edges_pos = np.argwhere(max_cutting_edges == cutting_edges)
    logger.info('Max number of cutting edges between clusters: %d' % (max_cutting_edges))
    logger.info('Clusters with max edges cutting: %s' % (max_cutting_edges_pos, ))

    # If we have more than one cluster with the same amount of cutting
    if len(max_cutting_edges_pos) > 1:
        logger.info('More than one cluster with max cutting edges')
        equal_partition = G.number_of_nodes()/final_k
        logger.info('Equal partition should be: %.3f' % (equal_partition))
        distances = []
        for max_edge in max_cutting_edges_pos:
            cluster_1_size = cluster_sizes[max_edge[0]]
            cluster_2_size = cluster_sizes[max_edge[1]]
            sizes_combined = cluster_1_size + cluster_2_size
            distance_ideal = abs(equal_partition - sizes_combined)
            logger.info('Cluster: %d. Size: %d' % (max_edge[0], cluster_1_size))
            logger.info('Cluster: %d. Size: %d' % (max_edge[1], cluster_2_size))
            logger.info('Distance from ideal: %d' % (distance_ideal))
            distances.append(distance_ideal)
        distances = np.array(distances)
        logger.info('Distance from each merging to ideal: %s' % (distances, ))
        idx_merge = np.argmin(distances)
        logger.info('Choosing to merge clusters in index: %d' % (idx_merge))
    else:
        # There is only one element with max cutting edges
        idx_merge = 0
    merge = max_cutting_edges_pos[idx_merge]

    new_node_clusters = {}
    for node, cluster in node_cluster.items():
        if node in new_node_clusters:
            raise ValueError('Something went wrong. Node classified twice')
        else:
            if cluster == merge[1]:
                new_node_clusters[node] = merge[0]
            else:
                new_node_clusters[node] = cluster
    logger.info('Merging finished')

    logger.info('Re-indexing of nodes in clusters')
    current_clusters = new_node_clusters.values()
    # Get the different values (clusters) within the dictionary
    current_clusters = list(set(current_clusters))
    re_index_node_clusters = {}
    for i, cluster in enumerate(current_clusters):
        for node, clus_node in new_node_clusters.items():
            if clus_node == cluster:
                if node in re_index_node_clusters:
                    raise ValueError('Something went wrong. Node classified twice')
                else:
                    re_index_node_clusters[node] = i
    logger.info('Re-indexing finished')

    return merge_clusters(G, final_k, re_index_node_clusters, num_clusters-1, merge_need-1, logger)


def multi_merger(G, k_eig, k, clustering, n, logger):
    logger.info('Merging clusters until getting K')
    num_k_eig = k_eig.shape[-1]
    logger.info('Number of eigenvectors received: %d' % (num_k_eig))

    logger.info('Using %s to make initial cluster for multi-merger' % (clustering))
    if (clustering == 'Kmeans'):
        # Cluster using k-means
        _, cluster_labels = cluster_k_means(k_eig, num_k_eig, logger)
    elif (clustering == "Gmm"):
        # Cluster using gmm
        cluster_labels = cluster_gmm(k_eig, num_k_eig, logger)
    elif (clustering == "Agglomerative"):
        cluster_labels = cluster_agglomerative(k_eig, num_k_eig, logger)
    elif (clustering == 'Kmeans_modified'):
        cluster_labels = cluster_k_means_modified(G, n, k_eig, num_k_eig, logger)
    else:
        raise ValueError('Cannot multi merge cluster with: %s' % (clustering))

    cluster_labels = correct_cluster_labels(G, cluster_labels)

    logger.info('Making dictionary with cluster labels')
    node_cluster = {}
    for node, label in enumerate(cluster_labels):
        if node in node_cluster:
            raise ValueError('Something went wrong. Node classified twice')
        else:
            node_cluster[node] = label
    logger.info('Number of clusters before merging: %d' % (num_k_eig))
    need_merge = num_k_eig - k
    logger.info('Number of merges needed: %d' % (need_merge))

    node_cluster = merge_clusters(G, k, node_cluster, num_k_eig, need_merge, logger)
    num_nodes = len(node_cluster.keys())
    logger.info('Number of nodes merged: %d' % (num_nodes))

    logger.info('Merging finished')
    logger.info('Changing dictionary with keys by cluster')

    cluster_node = {}
    for node, cluster in node_cluster.items():
        if cluster in cluster_node:
            cluster_node[cluster].append(node)
        else:
            cluster_node[cluster] = [node]

    clusters = cluster_node.keys()
    logger.info('Final number of clusters %d' % (len(clusters)))
    logger.info('Arranging the nodes and clusters')
    new_cluster_labels = np.zeros(num_nodes)
    for i, cluster in enumerate(clusters):
        nodes = cluster_node[cluster]
        for node in nodes:
            new_cluster_labels[node] = i
    logger.info('Finished arranging. Finished merging clusters')

    return new_cluster_labels
