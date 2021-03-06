# Script for Graph partioning into communities

Course project of Algorithmic Methods of Data Mining. Winter semester 2019 at Aalto University.

Developed by:
* Erick Gijon Leyva
* Alejandro Ponce de León Chávez

The purpose of this project is to use different algorithms to get ideal partioning on different graphs given a scoring function. The score for the partioning can be calculated as follows:

<img src="https://latex.codecogs.com/gif.latex?\phi(V_1,...,V_k)=\sum^k_{i=1}\frac{|E(V_i,&space;\bar{V_i})|}{|V_i|}" title="\phi(V_1,...,V_k)=\sum^k_{i=1}\frac{|E(V_i,&space;\bar{V_i})|}{|V_i|}"/>

Where Vi is a partition of the graph and E is the set of edges in Vi that have on endpoint in Vi and another one in Vi-bar.

### Useful links

Links to libraries used during the realization of the project

* [Graph](https://networkx.github.io/documentation/networkx-2.4/reference/introduction.html#graphs) - For constructring graphs using NetworkX package in python
* [Graph iteration and other methods](https://networkx.github.io/documentation/networkx-2.4/reference/classes/graph.html#methods) - For iterating graphs and other useful methods in networkx library
* [Laplacian Matrix](https://networkx.github.io/documentation/networkx-2.4/reference/linalg.html#module-networkx.linalg.laplacianmatrix) - For getting the Laplacian matrix of the graph
* [Eigenvalues and Eigenvectors](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html) - For getting the Eigenvalues and Eigenvectors of the Laplacian Matrix. Numpy function.
* [Eigenvalues and Eigenvectors of Real symmetric matrix](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html#numpy.linalg.eigh) - For getting the Eigenvalues and Eigenvectors of the Laplacian Matrix. Numpy function.
* [K-Eigenvalues and K-Eigenvectors of Real symmetric matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh) - For getting the K-eigenvalues and K-eigenvectors of the Laplacian Matrix. Numpy function.
* [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) - For computing K-means algorithm to cluster.
* [Logger](https://docs.python.org/3/library/logging.html) - For logging into console and to a file.
* [Argparse](https://docs.python.org/3/howto/argparse.https://docs.python.org/3/library/time.htmlhttps://docs.python.org/3/howto/argparse.html) - For timing the execution of the program and getting running machine time.

### Installation

Requires Python 3.

Recommended to use a virtual environment. Install the dependencies.

```sh
$ pip install -r requirements.txt
```
### Running

The script requires a graph as argument to start running and an algorithm

Available options

| Graph | Vertices | Edges | k |
| ------ | ------ | ------ | ------ |
| dummy | 12 | 12 | 2 |
| ca-GrQc | 4158 | 13428 | 2 |
| ca-HepTh | 8638 | 24827 | 20 |
| Oregon-1 | 10670 | 22002 | 5 |
| ca-HepPh | 11204 | 117649 | 25 |
| ca-AstroPh | 17903 | 197031 | 50 |
| ca-CondMat | 21363 | 91342 | 100 |
| soc-Epinions1 | 75879 | 405740 | 10 |
| web-NotreDame | 325729 | 1117563 | 20 |
| roadNet-CA | 1957027 | 2760388 | 50 |

Available algorithms:
- Unorm - Use un-normalized Laplacian and k clusters
- NormLap - Use normalized Laplacian and k clusters
- NormEig - Use normalized Laplacian, normalized rows of eigenvector matrix and k clusters
- HagenKahng - Only useful for k=2. Use only second eigenvector to cluster. Based on [New Spectral Methods for Ratio Cut Partitioning and Clustering](https://pdfs.semanticscholar.org/3627/8bf6919c6dced7d16dc0c02d725e1ed178f8.pdf) by Lars Hagen and Andrew B. Kahng
- Recursive

```sh
$ python main.py -g <graph> -a <algorithm>
```

Example:

```sh
$ python main.py -g ca-GrQc -a Recursive
```

Optionally also specify the logging level

```sh
$ python main.py -g ca-GrQc -l INFO
```

To get some help on how to run the script

```sh
$ python main.py -h
```

There is also a bash script to run an algorithm over all of the graphs.

```sh
./execute_algorithm.sh <algorithm>
```

Example:

```sh
./execute_algorithm.sh Recursive
```

### Preliminary results

Using K-Means for clustering and K-eigenvectors

| Graph | Unorm | NormLap | NormEig | Recursive | HagenKahng |
|---------------|---------|---------|---------|-----------|-----------|
| ca-GrQc | 0.0835 | 0.0838 | 0.362 | 0.2305 | 2.00 |
| ca-HepTh | 2.442 | 5.936 | 17.933 | 10.67 | N/A |
| Oregon-1 | 0.694 | 1.560 | 1.300 | 5.74 | N/A |
| ca-HepPh | 5.578 | 15.280 | 59.487 | 51.484 | N/A |
| ca-AstroPh | 15.563 | 40.994 | 221.916 | 73.273 | N/A |
| ca-CondMat | 30.180 | 68.223 | 215.75 | Error | N/A |
| soc-Epinions1 | 0.7548 | 1.085 | 10.292 | 42.05 | N/A |
| web-NotreDame | ? | 2702.41 | 0.115 | Error | N/A |
| roadNet-CA | ? | 4.4936 | 0.276 | ? | N/A |

Using Agglomerative for clustering and K-eigenvectors

| Graph | Unorm | NormLap | NormEig | Recursive | HagenKahng |
|---------------|---------|---------|---------|-----------|-----------|
| ca-GrQc | 0.0835 | 0.0838 | 0.0757 | 0.2305 | N/A |
| ca-HepTh | 2.426 | 5.836 | 15.231 | ? | N/A |
| Oregon-1 | 0.693 | 1.031 | 1.173 | ? | N/A |
| ca-HepPh | 5.181 | 15.953 | 45.731 | ? | N/A |
| ca-AstroPh | 14.881 | 43.256 | 174.766 | ? | N/A |
| ca-CondMat | 28.786 | 66.055 | 197.186 | ? | N/A |
| soc-Epinions1 | Error | Error | Error | ? | N/A |
| web-NotreDame | ? | Error | Error | ? | N/A |
| roadNet-CA | ? | Error | Error | ? | N/A |

Graph competition

| Graph | Personal Best | Min Top 10
|---------------|---------------|---------------|
| ca-GrQc | 0.0627 | 0.0627 |
| Oregon-1 | 0.6003 | 0.6392 |
| soc-Epinions1 | 0.7548 | 0.7548 |
| web-NotreDame | 0.1151 | 0.0778 |
| roadNet-CA | 0.2762 | 0.2663 |

### Elapsed times

Elapsed time for running each algorithm in each graph. In seconds

| Graph | Unorm | NormLap | NormEig | Recursive | HagenKahng |
|---------------|---------|---------|---------|-----------|-----------|
| ca-GrQc | 2.50 | 0.52 | 0.60 | 0.84 | 87.10 |
| ca-HepTh | 1.98 | 0.94 | 1.30 | 23.49 | N/A |
| Oregon-1 | 18 | 0.48 | 0.48 | 1.967 | N/A |
| ca-HepPh | 8.174 | 1.99 | 2.99 | 122.00 | N/A |
| ca-AstroPh | 259.43 | 6.57 | 8.80 | 167.93 | N/A |
| ca-CondMat | 377.83 | 27.25 | 33.21 | Error | N/A |
| soc-Epinions1 | 4505.20 | 8.89 | 8.83 | 59.40 | N/A |
| web-NotreDame | ? | 285.24 | 285.80 | Error | N/A |
| roadNet-CA | ? | 11929.53 | 12004.48 | ? | N/A |


### Resources

Links to useful documentation

* [Normalized cuts and image segmentation](https://www.cis.upenn.edu/~jshi/papers/pami_ncut.pdf)
* [Balanced k-Means and Min-Cut Clustering](https://arxiv.org/abs/1411.6235)
* [Balanced K-Means for Clustering](https://link.springer.com/chapter/10.1007/978-3-662-44415-3_4) - Mikko I. Malinen and Pasi Fränti
* [New Spectral Methods for Ratio Cut Partitioning and Clustering](https://pdfs.semanticscholar.org/3627/8bf6919c6dced7d16dc0c02d725e1ed178f8.pdf) - Lars Hagen and Andrew B. Kahng
* [Graph Cuts and Normalized Cuts](http://www.dccia.ua.es/~sco/Spectral/Lesson3_Cuts.pdf)
* [Graph Clustering](http://cs-people.bu.edu/evimaria/cs565/lect19-20.pdf)
* [A Tutorial on Spectral Clustering](http://ranger.uta.edu/~chqding/Spectral/spectralA.pdf) - Chris Ding
* [Same-size k-Means Variation](https://elki-project.github.io/tutorial/same-size_k_means)
* [Graph Clustering Using Ratio Cuts](https://www.cis.upenn.edu/~cis515/cis515-15-spectral-clust-chap5.pdf)
