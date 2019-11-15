from graph_filenames import graphs_files
import sys
import argparse
import numpy as np


# Parse script arguments
def parse_args(graph_names, args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", "-g", type=str, required=True,
                        help="Graph to execute the algorithm",
                        choices=graph_names),
    parser.add_argument("--verbose", "-v", help="Verbose output",
                        action="store_true")
    return parser.parse_args(args)


def get_adj_matrix(graph_file):
    lines = graph_file.split("\n")
    print(lines[0])


if __name__ == '__main__':
    args = parse_args(list(graphs_files.keys()))
    graph_file_content = ''
    with open(graphs_files[args.graph], 'r') as file:
        graph_file_content = file.read()
    get_adj_matrix(graph_file_content)
