from resources import (graphs_files, algorithms, clustering, OUT_FOLDER,
                       COMP_FOLDER)
import sys
import argparse
import networkx as nx
import numpy as np
import logging
import time
import os
from algorithms import (unorm, norm_lap, norm_eig, recursive, hagen_kahng,
                        score_function, norm_eig_col)

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
    # Use different cluster algorithms to run the script
    parser.add_argument("--cluster", "-c",
                        type=str, help="Indicate clustering", default="Kmeans",
                        choices=clustering)
    parser.add_argument("--nodes", "-n",
                        type=int, default=0,
                        help="Nodes to mergge in modified Kmans")
    # Use more than k eigenvectors to run the clustering
    parser.add_argument("--k_custom", "-k",
                        type=int, default=0,
                        help="Indicate a custom number of k to get eigenvectors")
    # Merge clusters
    parser.add_argument("--merge", "-m",
                        action="store_true",
                        help="If there is a k_custom then merge manually the extra clusters")
    # Argument to print to console
    parser.add_argument("--log", "-l",
                        type=str, help="Set logging level", default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    # Argument to name the log file
    parser.add_argument("--file", "-f",
                        type=str, help="Name of the log file",
                        default="graph.log")
    # Flag to dump the first eigenvector
    parser.add_argument("--dump",
                        action="store_true", help="Discard the first eigenvector")
    # Flag to force the re-calculate eigenvector and laplacian
    parser.add_argument("--no_cache",
                        action="store_true",
                        help="For recalculation of eigenvectos and Laplacian")
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
    # Getting the edges from file
    file_graph = [line.split(' ') for line in file_graph]
    G = nx.Graph()
    # Forcing correct node order in graph
    for i in range(graph_meta['vertices']):
        G.add_node(str(i))
    # Adding edges to graph
    G.add_edges_from(file_graph)

    return graph_meta, G


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
    str_time = time.strftime("%m-%d-%Y_%H_%M_%S", time.localtime())
    out_name = '%s_%s.output' % (g_meta['name'], str_time)

    logger.info('Returning string to write output file')

    return out_name, header + cluster_str


def run_algorithm(G, G_meta, algo, clustering, dump, cache, k, n, merge, logger):
    logger.info('Going to execute algorithm: %s' % (algo))
    if (algo == 'Unorm'):
        cluster_labels = unorm(G=G, G_meta=G_meta, clustering=clustering,
                               dump=dump, cache=cache, k=k, n=n, merge=merge,
                               logger=logger)
    elif (algo == 'NormLap'):
        cluster_labels = norm_lap(G=G, G_meta=G_meta, clustering=clustering,
                                  dump=dump, cache=cache, k=k, n=n, merge=merge,
                                  logger=logger)
    elif(algo == 'NormEig'):
        cluster_labels = norm_eig(G=G, G_meta=G_meta, clustering=clustering,
                                  dump=dump, cache=cache, k=k, n=n, merge=merge,
                                  logger=logger)
    elif(algo == 'NormEigCol'):
        cluster_labels = norm_eig_col(G=G, G_meta=G_meta, clustering=clustering,
                                      dump=dump, cache=cache, k=k, n=n, merge=merge,
                                      logger=logger)
    elif(algo == 'Recursive'):
        # Empty dictionary to track  labels
        k = G_meta['k']
        c = {}
        cluster_labels = recursive(G, k, c, clustering, n, logger)
        np_labels = np.zeros(len(list(G)))
        for i in range(k):
            for j in cluster_labels[i]:
                np_labels[int(j)] = i
        cluster_labels = np_labels
    elif(algo == 'HagenKahng'):
        cluster_labels = hagen_kahng(G, G_meta, cache, logger)
    logger.info('Algorithm execution finished: %s' % (algo))

    return cluster_labels


def main(logger):
    graph_file_content = ''
    with open(graphs_files[args.graph], 'r') as file:
        graph_file_content = file.read()
    # Get a graph object from the file content
    G_meta, G = get_graph(graph_file_content, logger)
    logger.info("Number of nodes: %d" % (G.number_of_nodes()))
    logger.info("Number of edges: %d" % (G.number_of_edges()))

    # Make a folder to store the eigenvectors
    os.makedirs(COMP_FOLDER, exist_ok=True)

    # In case there was a different number of k values specified
    k = G_meta['k']
    if args.k_custom != 0:
        k = args.k_custom

    cluster_labels = run_algorithm(G=G, G_meta=G_meta, algo=args.algo,
                                   clustering=args.cluster, dump=args.dump,
                                   cache=args.no_cache, k=k, n=args.nodes,
                                   merge=args.merge, logger=logger)

    # Getting the data to writhe to file
    out_name, out_str = output_file(G_meta, cluster_labels, logger)
    os.makedirs(OUT_FOLDER, exist_ok=True)
    out_path = OUT_FOLDER + '/' + out_name
    logger.info('Writing results to file: %s' % (out_path))
    with open(out_path, 'w') as file:
        file.write(out_str)
    end_time = time.time()
    logger.info('Finished execution. Elapsed time: %.10f sec' % (end_time - start_time))
    score = score_function(cluster_labels, G_meta['k'], G, logger)
    end_time_score = time.time()
    logger.info('Finished score execution. Elapsed time: %.10f sec' % (end_time_score - end_time))
    logger.info('Score obtained from clustering: %.10f' % (score))
    logger.info('************** Summary **************')
    logger.info('Graph: %s' % (G_meta['name']))
    logger.info('Algorithm used: %s' % (args.algo))
    logger.info('N to merge: %d' % (args.nodes))
    logger.info('K custom: %d' % (args.k_custom))
    logger.info('Dumping?: %s' % (args.dump))
    logger.info('Merging?: %s' % (args.merge))
    logger.info('Clustering algorithm used: %s' % (args.cluster))
    logger.info('Score of execution: %.10f' % (score))
    logger.info('Elapsed time of execution: %.10f' % (end_time - start_time))
    logger.info('Output file: %s' % (out_path))
    logger.info('*************************************')


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
    try:
        main(logger)
    except Exception:
        logger.exception("Fatal error in main loop")
