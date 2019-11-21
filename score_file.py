from resources import graphs_files
import sys
import argparse
import logging
import time
import networkx as nx
import numpy as np


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    # Argument to print to console
    parser.add_argument("--log", "-l",
                        type=str, help="Set logging level", default="DEBUG",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    # Argument to name the log file
    parser.add_argument("--file", "-f",
                        type=str, help="Name of the log file",
                        default="scoring.log")
    # Argument to specify outoput file
    parser.add_argument("--output", "-o",
                        type=str, help="Name of output file to calculate the score",
                        required=True)
    return parser.parse_args(args)


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


def get_output_labels(out_file, logger):
    logger.debug('Parsing output file')

    out_file_content = ''
    with open(out_file, 'r') as file:
        out_file_content = file.read()

    # Get file lines into list
    lines = out_file_content.split("\n")
    # Remove empty lines from list
    lines = list(filter(None, lines))

    # Get the metadata of output from first line
    file_header = lines[0]
    file_header = file_header.split(' ')
    out_meta = {
        'name': file_header[1],
        'vertices': int(file_header[2]),
        'edges': int(file_header[3]),
        'k': int(file_header[4])
    }
    logger.debug('Going to get score for graph: %s' % (out_meta['name']))

    # Construct the graph with the rest of lines
    logger.debug('Getting the clusters of the nodes')
    cluster_graph = lines[1:]
    cluster_graph = [line.split(' ') for line in cluster_graph]
    cluster_label = []
    for cluster_line in cluster_graph:
        cluster_label.append(int(cluster_line[1]))

    return out_meta, np.array(cluster_label)


if __name__ == '__main__':
    start_time = time.time()
    # Read arguments from console
    args = parse_args()
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
    header_out, cluster_labels = get_output_labels(args.output, logger)
    # Read from the text file
    logger.debug('Reading graph from file: %s' % (graphs_files[header_out['name']]))
    graph_file_content = ''
    with open(graphs_files[header_out['name']], 'r') as file:
        graph_file_content = file.read()
    # Get a graph object from the file content
    G_meta, G = get_graph(graph_file_content, logger)
    logger.debug("Number of nodes: %d" % (G.number_of_nodes()))
    logger.debug("Number of edges: %d" % (G.number_of_edges()))
    score = score_function(cluster_labels, G_meta['k'], G, logger)
    logger.debug('Score obtained from clustering: %.10f' % (score))
    end_time = time.time()
    logger.debug('Finished score execution. Elapsed time: %.10f sec' % (end_time - start_time))
