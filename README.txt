# Script for Graph partioning into communities

Course project of Algorithmic Methods of Data Mining. Winter semester 2019 at Aalto University.

Developed by:
* Alejandro Ponce de León Chávez
* Erick Gijon Leyva

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
- NormEigCol - Use normalized Laplacian, normalized rows and columns of eigenvector matrix and k clusters
- HagenKahng - Only useful for k=2. Use only second eigenvector to cluster
- Recursive

For a basic run:

```sh
$ python main.py -g <graph> -a <algorithm> 
```

Example:

```sh
$ python main.py -g ca-GrQc -a Recursive
```

To reproduce the results presented in the report, 
for example the best result on the graph ca-HepPh we have this in the report "NL Km 100 d m n 7"

```sh
$ python main.py -g ca-HepPh -a Kmeans_modified -k 100 -d -m -n 7
```

To get some help on how to run the script

```sh
$ python main.py -h
```

There is also a bash script to run all the algorithms over a graph, max_k is the number o eigenvectors to obtain and n_nodes the maximum number of nodes to merge

```sh
./execute_algorithm.sh <graph> <max_k> <n nodes>
```

Example:

```sh
./execute_algorithm.sh ca-GrQc
```