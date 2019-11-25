from resources import graphs_files, algorithms, clustering
import sys
import argparse
import networkx as nx
import numpy as np
import logging
import time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from scipy import sparse
import os


OUT_FOLDER = './outputs'


# Parse script arguments
def parse_args(graph_names, args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    # Needs a graph to execute this script
    parser.add_argument("--graph", "-g",
                        type=str, required=True,
                        help="Graph to execute the algorithm",
                        choices=graph_names)
    # Use different algorithms to run the script
    parser.add_argument("--algo", "-a",
                        type=str, help="Indicate normalization", default="Unorm",
                        choices=algorithms)
    # Use different algorithms to run the script
    parser.add_argument("--cluster", "-c",
                        type=str, help="Indicate clustering", default="Kmeans",
                        choices=clustering)
    # Argument to print to console
    parser.add_argument("--log", "-l",
                        type=str, help="Set logging level", default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    # Argument to name the log file
    parser.add_argument("--file", "-f",
                        type=str, help="Name of the log file",
                        default="graph.log")
    return parser.parse_args(args)


def get_graph(graph_file, logger):
    logger.info('Parsing graph file')
    # Get file lines into list
    lines = graph_file.split("\n")
    # Remove empty lines from list
    lines = list(filter(None, lines))

    # Get the metadata of graph from first line
    file_header = lines[0]
    file_header = file_header.split(' ')
    graph_meta = {
        'name': file_header[1],
        'vertices': int(file_header[2]),
        'edges': int(file_header[3]),
        'k': int(file_header[4])
    }

    # Construct the graph with the rest of lines
    logger.info('Constructing graph')
    file_graph = lines[1:]
    file_graph = [line.split(' ') for line in file_graph]
    G = nx.Graph()
    G.add_edges_from(file_graph)

    return graph_meta, G


def laplacian_and_k_eigenval_eigenvec(G, k, norm_decide, logger):
    # Get the Laplacian matrix from the graph
    #node_list = [str(i) for i in range(len(list(G)))]
    if('norm' in norm_decide):
        logger.info('Getting Normalized Laplacian matrix')
        #L = nx.normalized_laplacian_matrix(G, node_list)
        L = nx.normalized_laplacian_matrix(G)
    else:
        logger.info('Getting Laplacian matrix')
        L = nx.laplacian_matrix(G)
    L_double = L.asfptype()
    # Get the eigenvalues and eigenvectors
    logger.info('Getting eigenvalues and eigenvectors of Laplacian')
    # Note use of function eigsh over eig.
    # eigsh for real symmetric matrix and only k values
    eigenval, eigenvec = sparse.linalg.eigsh(L_double, which='SM', k=k, ncv=5*k)
    if (norm_decide == 'norm_eig'):
        logger.info('Normalizing eigenvec matrix')
        eigenvec = normalize(eigenvec, axis=1, norm='l2')
    logger.info('Finished. Returning eigenvalues, eigenvectors and Laplacian')
    return L, eigenval, eigenvec

def cluster_k_means(k_eig, k, logger):
    logger.info('Using k-means to cluster the vertices')
    kmeans = KMeans(n_clusters=k).fit(k_eig)
    logger.info('K-means finished. Returning the results')
    return kmeans.labels_

def cluster_gmm(k_eig, k, logger):
    logger.info('Using GMM to cluster the vertices')
    gmm = GaussianMixture(n_components = k).fit(k_eig)
    labels = gmm.predict(k_eig)
    logger.info('GMM finished. Returning the results')
    return labels


def output_file(g_meta, clustered, logger):
    logger.info('Preparing string to write output file')
    # Prepare the header of the output file
    header = '# %s %d %d %d\n' % (
                                    g_meta['name'],
                                    g_meta['vertices'],
                                    g_meta['edges'],
                                    g_meta['k']
                                )
    # Getting the rest of the lines of the output file
    # vertex1ID partition1ID
    cluster_str = ''
    for i, cluster in enumerate(clustered):
        cluster_str += '%d %d\n' % (i, cluster)

    # Constructing the output filename
    str_time = time.strftime("%m-%d-%Y_%H_%M", time.localtime())
    out_name = '%s_%s.output' % (g_meta['name'], str_time)

    logger.info('Returning string to write output file')

    return out_name, header + cluster_str


def hagen_kahng_ratio_cut(eigv_2, G, logger):
    logger.info('Executing Hagen Kahng algorithm')
    logger.debug('Sorting second eigenvector')
    # Getting the indexes for sorting the second eigenvector
    ordered_eigv = np.argsort(eigv_2)
    logger.debug(eigv_2[ordered_eigv])
    # Initialize an array to keep the results for every cut
    r_results = np.zeros(G.number_of_nodes()-1)
    # Make all possible cuts over the second eigenvector
    for i in range(len(r_results)):
        # Putting i nodes in the first cluster
        cluster_1_num = i+1
        # Putting V-i nodes in the second cluster
        cluster_2_num = G.number_of_nodes()-cluster_1_num
        logger.debug('Iteration of Hagen Kahng: %d' % (i))
        logger.debug('Nodes in Cluster 1: %d' % (cluster_1_num))
        logger.debug('Nodes in Cluster 2: %d' % (cluster_2_num))
        cluster_1 = np.zeros(cluster_1_num)
        cluster_2 = np.ones(cluster_2_num)
        clustered = np.append(cluster_1, cluster_2)
        # Ordering the cut according to the ordered eigenvector
        clustered = clustered[ordered_eigv]
        # Getting the score for that cut
        r_results[i] = score_function(clustered, k=2, G=G, logger=logger)
        logger.debug('Scoring function results: %.10f' % (r_results[i]))
    # Get the best score from all the cuts
    ideal_cut = np.argmin(r_results) + 1
    # Constructing again the way the cut was made
    clustered = np.append(np.zeros(ideal_cut), np.ones(G.number_of_nodes()-ideal_cut))
    clustered = clustered[ordered_eigv]
    logger.info('Best scoring function found Hagen Kahng: %.10f' % (r_results[ideal_cut-1]))
    logger.info('Cluster 1 size: %d' % (ideal_cut))
    logger.info('Cluster 2 size: %d' % (G.number_of_nodes()-ideal_cut))
    return clustered


def score_function(clustered, k, G, logger):
    equal_partition = G.number_of_nodes()/k
    logger.debug('Ideal balanced clusters: %.10f' % (equal_partition))
    logger.debug('Getting score for the clustering')
    k_score = []
    # Iterate over the k clusters
    for i in range(k):
        # Get the nodes that were classified as the cluster k
        indexes = np.where(clustered == i)[0]
        cluster_size = len(indexes)
        logger.debug('Cluster: %d. Number of nodes: %d' % (i, cluster_size))
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
        logger.debug('Cluster: %d. Edges cutting clusters: %d' % (i, edge_diff_cluster))
        logger.debug('Cluster: %d. Score: %.10f' % (i, cluster_score))
        k_score.append(cluster_score)
    return sum(k_score)


def unorm(G, k,  clustering, PCA,logger):
    # Get Laplacian, k eigenvalues and eigenvectors of it
    _, k_eigenval, k_eigenvec = laplacian_and_k_eigenval_eigenvec(G, k + 1, 'u', logger)
    k_eigenvec = k_eigenvec[:,1:] 
    logger.debug("Shape of K eigenvector matrix: %s" % (k_eigenvec.shape, ))
    logger.debug('K-Eigenvectors')
    logger.debug(k_eigenvec)
    logger.debug('K-Eigenvalues')
    logger.debug(k_eigenval)
    if (clustering == 'Kmeans'):
        # Cluster using k-means
        cluster_labels = cluster_k_means(k_eigenvec, k, logger)
    if (clustering == "Gmm"):
        #Cluster using gmm
        cluster_labels = cluster_gmm(k_eigenvec, k, logger)
    all_nodes = list(G)
    np_labels = np.zeros(len(all_nodes))
    for i in range(len(all_nodes)):
        np_labels[int(all_nodes[i])] = cluster_labels[i]
    return np_labels


def norm_lap(G, k, clustering, PCA, logger):
    # Get Laplacian, k eigenvalues and eigenvectors of it
    _, k_eigenval, k_eigenvec = laplacian_and_k_eigenval_eigenvec(G, k + 1, 'norm', logger)
    k_eigenvec = k_eigenvec[:,1:]
    logger.debug("Shape of K eigenvector matrix: %s" % (k_eigenvec.shape, ))
    logger.debug('K-Eigenvectors')
    logger.debug(k_eigenvec)
    logger.debug('K-Eigenvalues')
    logger.debug(k_eigenval)
    if (clustering == 'Kmeans'):
        # Cluster using k-means
        cluster_labels = cluster_k_means(k_eigenvec, k, logger)
    if (clustering == "Gmm"):
        #Cluster using gmm
        cluster_labels = cluster_gmm(k_eigenvec, k, logger)

    all_nodes = list(G)
    np_labels = np.zeros(len(all_nodes))
    for i in range(len(all_nodes)):
        np_labels[int(all_nodes[i])] = cluster_labels[i]
    return np_labels


def norm_eig(G, k, clustering, PCA, logger):
    # Get Laplacian, k eigenvalues and eigenvectors of it
    _, k_eigenval, k_eigenvec = laplacian_and_k_eigenval_eigenvec(G, k + 1, 'norm_eig', logger)
    k_eigenvec = k_eigenvec[:,1:]
    logger.debug("Shape of K eigenvector matrix: %s" % (k_eigenvec.shape, ))
    logger.debug('K-Eigenvectors')
    logger.debug(k_eigenvec)
    logger.debug('K-Eigenvalues')
    logger.debug(k_eigenval)
    if (clustering == 'Kmeans'):
        # Cluster using k-means
        cluster_labels = cluster_k_means(k_eigenvec, k, logger)
    if (clustering == "Gmm"):
        #Cluster using gmm
        cluster_labels = cluster_gmm(k_eigenvec, k, logger)

    all_nodes = list(G)
    np_labels = np.zeros(len(all_nodes))
    for i in range(len(all_nodes)):
        np_labels[int(all_nodes[i])] = cluster_labels[i]
    return np_labels

def recursive(G, k, c, clustering, PCA, logger):
    if (k >= 2):
        # Get Laplacian, 2 eigenvalues and eigenvectors
        _, _, k_eigenvec = laplacian_and_k_eigenval_eigenvec(G, 2, 'norm', logger)
        logger.debug("Shape of K eigenvector matrix: %s" % (k_eigenvec.shape, ))
        # Cluster using k-means and the second smallest eigenvector
        eigenvec_2 = k_eigenvec[:,1].reshape(-1,1)
        #eigenvec_2 = k_eigenvec
        if (clustering == 'Kmeans'):
            # Cluster using k-means
            cluster_labels = cluster_k_means(eigenvec_2, 2, logger)
        if (clustering == "Gmm"):
            #Cluster using gmm
            cluster_labels = cluster_gmm(eigenvec_2, 2, logger)
        #cluster_labels = cluster_gmm(eigenvec_2, 2, logger)
        # Nodes of the biggest cluster
        logger.debug("Graph partition")
        all_nodes = list(G)
        n = len(all_nodes)
        b_cluster = sum(cluster_labels)
        logger.info('Remaining iterations: %d.' % (k-1))
        if (b_cluster > n/2):
            indicator = 1
            indicator2 = 0
        else:
            indicator = 0
            indicator2 = 1
        nodes = [all_nodes[i] for i, label in enumerate(cluster_labels) if label==indicator]
        accepted_cluster = [all_nodes[i] for i, label in enumerate(cluster_labels) if label==indicator2]
        subgraph = G.subgraph(nodes)
        c[k-1] = accepted_cluster
        return recursive(subgraph, k-1, c, clustering, PCA, logger)
        #return c
    c[k-1] = list(G)
    return c

def hagen_kahng(G, k, logger):
    # Throws an execption if k!=2
    if k != 2:
        raise ValueError(('Hagen Kahng algorithm only works with k=2.'
                          'Trying to execute k=%d') % (k))
    # Get the Unormalized Laplacian matrix
    _, k_eigenval, k_eigenvec = laplacian_and_k_eigenval_eigenvec(G, k, 'u', logger)
    logger.debug("Shape of K eigenvector matrix: %s" % (k_eigenvec.shape, ))
    logger.debug('K-Eigenvectors')
    logger.debug(k_eigenvec)
    logger.debug('K-Eigenvalues')
    logger.debug(k_eigenval)
    # Getting only the second eigenvector
    logger.info("Getting only second eigenvector")
    eigv_2 = k_eigenvec[:, 1]
    logger.debug(eigv_2)
    # Executing the Hagen Kahng algorithm
    cluster_labels = hagen_kahng_ratio_cut(eigv_2, G, logger)

    return cluster_labels

def run_algorithm(G, k, algo, clustering, logger):
    logger.info('Going to execute algorithm: %s' % (algo))
    if (algo == 'Unorm'):
        cluster_labels = unorm(G, k, clustering, False, logger)
    elif (algo == 'NormLap'):
        cluster_labels = norm_lap(G, k, clustering, False, logger)
    elif(algo == 'NormEig'):
        cluster_labels = norm_eig(G, k, clustering, False, logger)
    elif(algo == 'Recursive'):
        #Empty dictionary to track  labels
        c={}
        cluster_labels = recursive(G, k, c , clustering, False, logger)
        np_labels = np.zeros(len(list(G)))
        for i in range(k):
            for j in cluster_labels[i]:
                np_labels[int(j)]=i
        cluster_labels = np_labels
    elif(algo == 'HagenKahng'):
        cluster_labels = hagen_kahng(G, k, logger)
    logger.info('Algorithm execution finished: %s' % (algo))

    return cluster_labels


if __name__ == '__main__':
    start_time = time.time()
    # Read arguments from console
    args = parse_args(list(graphs_files.keys()))
    # Get a logger of the events
    numeric_log_level = getattr(logging, args.log, None)
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S %p',
        level=numeric_log_level,
        handlers=[
            logging.FileHandler(args.file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info('Logger ready. Logging to file: %s' % (args.file))
    # Read from the text file
    logger.info('Reading graph from file: %s' % (graphs_files[args.graph]))
    graph_file_content = ''
    with open(graphs_files[args.graph], 'r') as file:
        graph_file_content = file.read()
    # Get a graph object from the file content
    G_meta, G = get_graph(graph_file_content, logger)
    logger.info("Number of nodes: %d" % (G.number_of_nodes()))
    logger.info("Number of edges: %d" % (G.number_of_edges()))

    cluster_labels = run_algorithm(G, G_meta['k'], args.algo, args.cluster, logger)

    # Getting the data to writhe to file
    out_name, out_str = output_file(G_meta, cluster_labels, logger)
    os.makedirs(OUT_FOLDER, exist_ok=True)
    out_path = OUT_FOLDER + '/' + out_name
    logger.info('Writing results to file: %s' % (out_name))
    with open(out_path, 'w') as file:
        file.write(out_str)
    end_time = time.time()
    logger.info('Finished execution. Elapsed time: %.10f sec' % (end_time - start_time))
    score = score_function(cluster_labels, G_meta['k'], G, logger)
    logger.info('Score obtained from clustering: %.10f' % (score))
    end_time_score = time.time()
    logger.info('Finished score execution. Elapsed time: %.10f sec' % (end_time_score - end_time))
