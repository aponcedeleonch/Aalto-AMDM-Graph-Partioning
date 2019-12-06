import numpy as np

graphs_files = {
    'ca-AstroPh': 'graphs_processed/ca-AstroPh.txt',
    'ca-CondMat': 'graphs_processed/ca-CondMat.txt',
    'ca-GrQc': 'graphs_processed/ca-GrQc.txt',
    'ca-HepPh': 'graphs_processed/ca-HepPh.txt',
    'ca-HepTh': 'graphs_processed/ca-HepTh.txt',
    'Oregon-1': 'graphs_processed/Oregon-1.txt',
    'roadNet-CA': 'graphs_processed/roadNet-CA.txt',
    'soc-Epinions1': 'graphs_processed/soc-Epinions1.txt',
    'web-NotreDame': 'graphs_processed/web-NotreDame.txt',
    'dummy': 'graphs_processed/dummy.txt'
}

algorithms = ['HagenKahng', 'Recursive', 'LaplacianEigenvectors']
laplacians = ['Unorm', 'Norm']
eigenvectors = ['None', 'Norm', 'NormCol']
clustering = ['Kmeans', 'Gmm', 'Agglomerative', 'Kmeans_modified']
merges = ['edges', 'size']

OUT_FOLDER = './outputs'
COMP_FOLDER = './computed'


def score_function(clustered, k, G, logger):
    equal_partition = G.number_of_nodes()/k
    logger.info('Ideal balanced clusters: %.10f' % (equal_partition))
    logger.debug('Getting score for the clustering')
    k_score = []
    # Iterate over the k clusters
    for i in range(k):
        # Get the nodes that were classified as the cluster k
        indexes = np.where(clustered == i)[0]
        cluster_size = len(indexes)
        logger.info('Cluster: %d. Number of nodes: %d' % (i, cluster_size))
        edge_diff_cluster = 0
        # Iterate over the nodes in cluster k
        for idx in indexes:
            node_cluster = clustered[idx]
            # Iterate over the neighbors of the node
            for neigh in G[str(idx)]:
                neigh_cluster = clustered[int(neigh)]
                # Check if the neighbor is in the same cluster as current node
                if node_cluster != neigh_cluster:
                    edge_diff_cluster += 1
        # Get the score for the cluster k. Store it
        cluster_score = edge_diff_cluster/cluster_size
        logger.info('Cluster: %d. Edges cutting clusters: %d' % (i, edge_diff_cluster))
        logger.debug('Cluster: %d. Score: %.10f' % (i, cluster_score))
        k_score.append(cluster_score)
    return sum(k_score)


def correct_cluster_labels(G, cluster_labels):
    all_nodes = list(G)
    np_labels = np.zeros(len(all_nodes))
    for i in range(len(all_nodes)):
        np_labels[int(all_nodes[i])] = cluster_labels[i]
    return np_labels
