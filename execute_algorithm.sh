#!/bin/sh

GRAPH=$1
KMAX=$2
N=$3

echo "Started execution of graph $GRAPH with algorithm NormEig, Kmeans and k=$KMAX\n"
python main.py -g $GRAPH -a NormEig -c Kmeans -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEig, Kmeans and k=$KMAX\n\n"

echo "Started execution of graph $GRAPH with algorithm NormLap, Kmeans and k=$KMAX\n"
python main.py -g $GRAPH -a NormLap -c Kmeans -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormLap, Kmeans and k=$KMAX\n\n"

echo "Started execution of graph $GRAPH with algorithm NormEigCol, Kmeans and k=$KMAX\n"
python main.py -g $GRAPH -a NormEigCol -c Kmeans -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEigCol, Kmeans and k=$KMAX\n\n"


echo "Started execution of graph $GRAPH with algorithm NormEig\n"
python main.py -g $GRAPH -a NormEig -c Kmeans -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEig\n\n"

echo "Started execution of graph $GRAPH with algorithm NormLap, Kmeans\n"
python main.py -g $GRAPH -a NormLap -c Kmeans -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormLap, Kmeans\n\n"

echo "Started execution of graph $GRAPH with algorithm NormEigCol, Kmeans\n"
python main.py -g $GRAPH -a NormEigCol -c Kmeans -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEigCol, Kmeans\n\n"

echo "====="

echo "Started execution of graph $GRAPH with algorithm NormEig, Gmm and k=$KMAX\n"
python main.py -g $GRAPH -a NormEig -c Gmm -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEig, Gmm and k=$KMAX\n\n"

echo "Started execution of graph $GRAPH with algorithm NormLap, Gmm and k=$KMAX\n"
python main.py -g $GRAPH -a NormLap -c Gmm -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormLap, Gmm and k=$KMAX\n\n"

echo "Started execution of graph $GRAPH with algorithm NormEigCol, Gmm and k=$KMAX\n"
python main.py -g $GRAPH -a NormEigCol -c Gmm -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEigCol, Gmm and k=$KMAX\n\n"


echo "Started execution of graph $GRAPH with algorithm NormEig\n"
python main.py -g $GRAPH -a NormEig -c Gmm -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEig\n\n"

echo "Started execution of graph $GRAPH with algorithm NormLap, Gmm\n"
python main.py -g $GRAPH -a NormLap -c Gmm -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormLap, Gmm\n\n"

echo "Started execution of graph $GRAPH with algorithm NormEigCol, Gmm\n"
python main.py -g $GRAPH -a NormEigCol -c Gmm -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEigCol, Gmm\n\n"

echo "====="

echo "Started execution of graph $GRAPH with algorithm NormEig, Agglomerative and k=$KMAX\n"
python main.py -g $GRAPH -a NormEig -c Agglomerative -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEig, Agglomerative and k=$KMAX\n\n"

echo "Started execution of graph $GRAPH with algorithm NormLap, Agglomerative and k=$KMAX\n"
python main.py -g $GRAPH -a NormLap -c Agglomerative -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormLap, Agglomerative and k=$KMAX\n\n"

echo "Started execution of graph $GRAPH with algorithm NormEigCol, Agglomerative and k=$KMAX\n"
python main.py -g $GRAPH -a NormEigCol -c Agglomerative -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEigCol, Agglomerative and k=$KMAX\n\n"


echo "Started execution of graph $GRAPH with algorithm NormEig\n"
python main.py -g $GRAPH -a NormEig -c Agglomerative -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEig\n\n"

echo "Started execution of graph $GRAPH with algorithm NormLap, Agglomerative\n"
python main.py -g $GRAPH -a NormLap -c Agglomerative -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormLap, Agglomerative\n\n"

echo "Started execution of graph $GRAPH with algorithm NormEigCol, Agglomerative\n"
python main.py -g $GRAPH -a NormEigCol -c Agglomerative -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEigCol, Agglomerative\n\n"

echo "====="

echo "Started execution of graph $GRAPH with algorithm NormEig\n"
python main.py -g $GRAPH -a NormEig -c Kmeans_modified -n $N -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEig\n\n"

echo "Started execution of graph $GRAPH with algorithm NormLap, Kmeans_modified\n"
python main.py -g $GRAPH -a NormLap -c Kmeans_modified -n $N -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormLap, Kmeans_modified\n\n"

echo "Started execution of graph $GRAPH with algorithm NormEigCol, Kmeans_modified\n"
python main.py -g $GRAPH -a NormEigCol -c Kmeans_modified -n $N -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEigCol, Kmeans_modified\n\n"

echo "====="

echo "Started execution of graph $GRAPH with algorithm NormEig, Kmeans and k=$KMAX\n"
python main.py -g $GRAPH -a Unorm -c Kmeans -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm Unorm, Kmeans and k=$KMAX\n\n"

echo "Started execution of graph $GRAPH with algorithm Unorm\n"
python main.py -g $GRAPH -a Unorm -c Kmeans -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm Unorm\n\n"

echo "Started execution of graph $GRAPH with algorithm Unorm, Gmm and k=$KMAX\n"
python main.py -g $GRAPH -a Unorm -c Gmm -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm Unorm, Gmm and k=$KMAX\n\n"

echo "Started execution of graph $GRAPH with algorithm Unorm\n"
python main.py -g $GRAPH -a Unorm -c Gmm -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm Unorm\n\n"

echo "Started execution of graph $GRAPH with algorithm Unorm\n"
python main.py -g $GRAPH -a Unorm -c Agglomerative -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm Unorm\n\n"

echo "Started execution of graph $GRAPH with algorithm NormEig, Agglomerative and k=$KMAX\n"
python main.py -g $GRAPH -a NormEig -c Agglomerative -k $KMAX -m -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm NormEig, Agglomerative and k=$KMAX\n\n"

echo "====="

echo "Started execution of graph $GRAPH with algorithm NormEig, Kmeans and k=$KMAX\n"
python main.py -g $GRAPH -a Unorm -c Kmeans_modified -n $N -f $GRAPH.log
echo "\nFinished execution of graph $GRAPH with algorithm Unorm, Kmeans and k=$KMAX\n\n"

