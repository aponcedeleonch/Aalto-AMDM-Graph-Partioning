from graph_filenames import graphs_files
import sys
import argparse


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", "-g", type=str, required=True,
                        help="Graph to execute the algorithm")
    parser.add_argument("--verbose", "-v", help="Verbose output",
                        action="store_true")
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    if args.verbose:
        print(args.graph)
    else:
        print("no verbose")
