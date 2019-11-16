# Graph partioning into communities

Course project of Algorithmic Methods of Data Mining. Winter semester 2019 at Aalto University.

Developed by:
* Erick Gijon Leyva
* Alejandro Ponce de León Chávez

### Useful links

Links to libraries used during the realization of the project

* [Graph](https://networkx.github.io/documentation/networkx-1.10/reference/classes.graph.html) - For constructring graphs using NetworkX package in python
* [Laplacian Matrix](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.linalg.laplacianmatrix.laplacian_matrix.html) - For getting the Laplacian matrix of the graph
* [Eigenvalues and Eigenvectors](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html) - For getting the Eigenvalues and Eigenvectors of the Laplacian Matrix. Numpy function.
* [Logger](https://docs.python.org/3/library/logging.html) - For logging into console and to a file.
* [Argparse](https://docs.python.org/3/howto/argparse.html) - For parsing arguments from the command line.

### Installation

Requires Python 3.

Recommended to use a virtual environment. Install the dependencies.

```sh
$ pip install requirements.txt
```
### Running

The script requires a graph as argument to start running

Available options

| Graph | Vertices | Edges | k |
| ------ | ------ | ------ | ------ |
| ca-AstroPh | 17903 | 197031 | 50 |
| ca-CondMat | 21363 | 91342 | 100 |
| ca-GrQc | 4158 | 13428 | 2 |
| ca-HepPh | 11204 | 117649 | 25 |
| ca-HepTh | 8638 | 24827 | 20 |
| Oregon-1 | 10670 | 22002 | 5 |
| roadNet-CA | 1957027 | 2760388 | 50 |
| soc-Epinions1 | 75879 | 405740 | 10 |
| web-NotreDame | 325729 | 1117563 | 20 |

Example

```sh
$ python main.py -g ca-GrQc
```

Optionally also specify the logging level

```sh
$ python main.py -g ca-GrQc -l INFO
```

To get some help on how to run the script

```sh
$ python main.py -h
```
