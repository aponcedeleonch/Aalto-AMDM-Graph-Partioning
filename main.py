from graph_filenames import graphs_files
import sys
import argparse
import networkx as nx
import numpy as np
import logging
import time


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
        'vertices': file_header[2],
        'edges': file_header[3],
        'k': file_header[4]
    }

    # Construct the graph with the rest of lines
    logger.debug('Constructing graph')
    file_graph = lines[1:]
    file_graph = [line.split(' ') for line in file_graph]
    G = nx.Graph()
    G.add_edges_from(file_graph)

    return graph_meta, G


def laplacian_and_eigenvalues(G, logger):
    # Get the Laplacian matrix from the graph
    logger.debug('Getting Laplacian matrix')
    L = nx.laplacian_matrix(G)
    L_numpy = L.todense()
    # Get the eigenvalues and eigenvectors
    logger.debug('Getting eigenvalues and eigenvectors of Laplacian')
    # Note use of function eigh over eig.
    # eigh for real symmetric matrix
    eigenval, eigenvec = np.linalg.eigh(L_numpy)
    logger.debug('Finished. Returning eigenvalues, eigenvectors and Laplacian')
    return L, eigenval, eigenvec


if __name__ == '__main__':
    # Read arguments from console
    args = parse_args(list(graphs_files.keys()))
    # Get a logger of the events
    numeric_log_level = getattr(logging, args.log, None)
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=numeric_log_level,
        handlers=[
            logging.FileHandler("graph.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    # Read from the text file
    graph_file_content = ''
    with open(graphs_files[args.graph], 'r') as file:
        graph_file_content = file.read()
    # Get a graph object from the file content
    G_meta, G = get_graph(graph_file_content, logger)
    start_time = time.time()
    L, eigenval, eigenvec = laplacian_and_eigenvalues(G, logger)
