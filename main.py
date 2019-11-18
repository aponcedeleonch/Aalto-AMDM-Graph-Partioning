from graph_filenames import graphs_files
import sys
import argparse
import networkx as nx
import numpy as np
import logging
import time
from sklearn.cluster import KMeans
from scipy import sparse


# Parse script arguments
def parse_args(graph_names, args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    # Needs a graph to execute this script
    parser.add_argument("--graph", "-g",
                        type=str, required=True,
                        help="Graph to execute the algorithm",
                        choices=graph_names)
    # Argument to print to console
    parser.add_argument("--log", "-l",
                        type=str, help="Set logging level", default="DEBUG",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    # Argument to name the log file
    parser.add_argument("--file", "-f",
                        type=str, help="Name of the log file",
                        default="graph.log")
    return parser.parse_args(args)


def get_graph(graph_file, logger):
    logger.debug('Parsing graph file')
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
    logger.debug('Constructing graph')
    file_graph = lines[1:]
    file_graph = [line.split(' ') for line in file_graph]
    G = nx.Graph()
    G.add_edges_from(file_graph)

    return graph_meta, G


def laplacian_and_k_eigenval_eigenvec(G, k, logger):
    # Get the Laplacian matrix from the graph
    logger.debug('Getting Laplacian matrix')
    L = nx.laplacian_matrix(G)
    L_double = L.asfptype()
    # Get the eigenvalues and eigenvectors
    logger.debug('Getting eigenvalues and eigenvectors of Laplacian')
    # Note use of function eigsh over eig.
    # eigsh for real symmetric matrix and only k values
    eigenval, eigenvec = sparse.linalg.eigsh(L_double, k=k)
    logger.debug('Finished. Returning eigenvalues, eigenvectors and Laplacian')
    return L, eigenval, eigenvec


def cluster_k_means(k_eig, k, logger):
    logger.debug('Using k-means to cluster the vertices')
    kmeans = KMeans(n_clusters=k).fit(k_eig)
    logger.debug('K-means finished. Returning the results')
    return kmeans.labels_


def output_file(g_meta, clustered, logger):
    logger.debug('Preparing string to write output file')
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
    out_name = '%s_output_%s.txt' % (g_meta['name'], str_time)

    return out_name, header + cluster_str


def score_function(clustered, k, G, logger):
    equal_partition = G.number_of_nodes()/k
    ideal_score = (1/equal_partition)*k
    logger.debug('Ideal score for clustering: %.10f' % (ideal_score))
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
    logger.debug('Logger ready. Logging to file: %s' % (args.file))
    # Read from the text file
    logger.debug('Reading graph from file: %s' % (graphs_files[args.graph]))
    graph_file_content = ''
    with open(graphs_files[args.graph], 'r') as file:
        graph_file_content = file.read()
    # Get a graph object from the file content
    G_meta, G = get_graph(graph_file_content, logger)
    # Get Laplacian, k eigenvalues and eigenvectors of it
    L, k_eigenval, k_eigenvec = laplacian_and_k_eigenval_eigenvec(G, G_meta['k'], logger)
    logger.debug("Shape of K eigenvector matrix: %s" % (k_eigenvec.shape, ))
    # Cluster using k-means
    cluster_labels = cluster_k_means(k_eigenvec, G_meta['k'], logger)
    # Getting the data to writhe to file
    out_name, out_str = output_file(G_meta, cluster_labels, logger)
    logger.debug('Writing results to file: %s' % (out_name))
    with open(out_name, 'w') as file:
        file.write(out_str)
    end_time = time.time()
    logger.debug('Finished execution. Elapsed time: %.10f sec' % (end_time - start_time))
    score = score_function(cluster_labels, G_meta['k'], G, logger)
    logger.debug('Score obtained from clustering: %.10f' % (score))
    end_time_score = time.time()
    logger.debug('Finished score execution. Elapsed time: %.10f sec' % (end_time_score - end_time))
