# Script for Graph partioning into communities

Course project of Algorithmic Methods of Data Mining. Winter semester 2019 at Aalto University.

Developed by:
* Erick Gijon Leyva
* Alejandro Ponce de León Chávez

### Useful links

Links to libraries used during the realization of the project

* [Graph](https://networkx.github.io/documentation/networkx-2.4/reference/introduction.html#graphs) - For constructring graphs using NetworkX package in python
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
$ pip install requirements.txt
```
### Running

The script requires a graph as argument to start running

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
### Preliminary results

| Graph | NormLap | NormEig | Recursive |
|---------------|---------|---------|-----------|
| ca-GrQc | ? | ? | 0.2305 |
| ca-HepTh | 100.04 | ? | ? |
| Oregon-1 | ? | ? | 5.74 |
| ca-HepPh | ? | 494.67 | ? |
| ca-AstroPh | ? | 208 | ? |
| ca-CondMat | 207.45 | ? | ? |
| soc-Epinions1 | ? | ? | ? |
| web-NotreDame | ? | ? | ? |
| roadNet-CA | ? | ? | ? |

### Resources

Links to useful documentation

* [Normalized cuts and image segmentation](https://www.cis.upenn.edu/~jshi/papers/pami_ncut.pdf)
* [Balanced k-Means and Min-Cut Clustering](https://arxiv.org/abs/1411.6235)
* [Balanced K-Means for Clustering](https://link.springer.com/chapter/10.1007/978-3-662-44415-3_4)
* [New Spectral Methods for Ratio Cut Partitioning and Clustering](https://pdfs.semanticscholar.org/3627/8bf6919c6dced7d16dc0c02d725e1ed178f8.pdf)
* [Graph Cuts and Normalized Cuts](http://www.dccia.ua.es/~sco/Spectral/Lesson3_Cuts.pdf)
* [Graph Clustering](http://cs-people.bu.edu/evimaria/cs565/lect19-20.pdf)
* [A Tutorial on Spectral Clustering](http://ranger.uta.edu/~chqding/Spectral/spectralA.pdf)
* [Same-size k-Means Variation](https://elki-project.github.io/tutorial/same-size_k_means)
* [Graph Clustering Using Ratio Cuts](https://www.cis.upenn.edu/~cis515/cis515-15-spectral-clust-chap5.pdf)
