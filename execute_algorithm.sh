#!/bin/sh

GRAPH=$1
KMAX=$2
N=$3

now=$(date +"%T")
echo $now
echo "Executing all with K means and merge\n"

python main.py -g $GRAPH -a NormEig -c Kmeans -k $KMAX -m -f $GRAPH.log
python main.py -g $GRAPH -a NormLap -c Kmeans -k $KMAX -m -f $GRAPH.log
python main.py -g $GRAPH -a NormEigCol -c Kmeans -k $KMAX -m -f $GRAPH.log

now=$(date +"%T")
echo $now
echo "Executing all with K means\n"

python main.py -g $GRAPH -a NormEig -c Kmeans -f $GRAPH.log
python main.py -g $GRAPH -a NormLap -c Kmeans -f $GRAPH.log
python main.py -g $GRAPH -a NormEigCol -c Kmeans -f $GRAPH.log

now=$(date +"%T")
echo $now
echo "Executing all with Gmm and merge\n"

python main.py -g $GRAPH -a NormEig -c Gmm -k $KMAX -m -f $GRAPH.log
python main.py -g $GRAPH -a NormLap -c Gmm -k $KMAX -m -f $GRAPH.log
python main.py -g $GRAPH -a NormEigCol -c Gmm -k $KMAX -m -f $GRAPH.log

now=$(date +"%T")
echo $now
echo "Executing all with Gmm\n"

python main.py -g $GRAPH -a NormEig -c Gmm -f $GRAPH.log
python main.py -g $GRAPH -a NormLap -c Gmm -f $GRAPH.log
python main.py -g $GRAPH -a NormEigCol -c Gmm -f $GRAPH.log

now=$(date +"%T")
echo $now
echo "Executing all with Agglomerative and merge\n"

python main.py -g $GRAPH -a NormEig -c Agglomerative -k $KMAX -m -f $GRAPH.log
python main.py -g $GRAPH -a NormLap -c Agglomerative -k $KMAX -m -f $GRAPH.log
python main.py -g $GRAPH -a NormEigCol -c Agglomerative -k $KMAX -m -f $GRAPH.log

now=$(date +"%T")
echo $now
echo "Executing all with Agglomerative\n"

python main.py -g $GRAPH -a NormEig -c Agglomerative -f $GRAPH.log
python main.py -g $GRAPH -a NormLap -c Agglomerative -f $GRAPH.log
python main.py -g $GRAPH -a NormEigCol -c Agglomerative -f $GRAPH.log

now=$(date +"%T")
echo $now
echo "Executing all with K means modified and merge\n"

python main.py -g $GRAPH -a NormEig -c Kmeans_modified -k $KMAX -m -n $N -f $GRAPH.log
python main.py -g $GRAPH -a NormLap -c Kmeans_modified -k $KMAX -m -n $N -f $GRAPH.log
python main.py -g $GRAPH -a NormEigCol -c Kmeans_modified -k $KMAX -m -n $N -f $GRAPH.log

now=$(date +"%T")
echo $now
echo "Executing all with K means modified\n"

python main.py -g $GRAPH -a NormEig -c Kmeans_modified -n $N -f $GRAPH.log
python main.py -g $GRAPH -a NormLap -c Kmeans_modified -n $N -f $GRAPH.log
python main.py -g $GRAPH -a NormEigCol -c Kmeans_modified -n $N -f $GRAPH.log

now=$(date +"%T")
echo $now
echo "Executing all with K means and merge and dumping\n"

python main.py -g $GRAPH -a NormEig -c Kmeans -k $KMAX -m -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormLap -c Kmeans -k $KMAX -m -f $GRAPH.log  --dump
python main.py -g $GRAPH -a NormEigCol -c Kmeans -k $KMAX -m -f $GRAPH.log  --dump

now=$(date +"%T")
echo $now
echo "Executing all with K means and dumping\n"

python main.py -g $GRAPH -a NormEig -c Kmeans -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormLap -c Kmeans -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormEigCol -c Kmeans -f $GRAPH.log --dump

now=$(date +"%T")
echo $now
echo "Executing all with Gmm and merge and dumping\n"

python main.py -g $GRAPH -a NormEig -c Gmm -k $KMAX -m -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormLap -c Gmm -k $KMAX -m -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormEigCol -c Gmm -k $KMAX -m -f $GRAPH.log --dump

now=$(date +"%T")
echo $now
echo "Executing all with Gmm and dumping\n"

python main.py -g $GRAPH -a NormEig -c Gmm -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormLap -c Gmm -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormEigCol -c Gmm -f $GRAPH.log --dump

now=$(date +"%T")
echo $now
echo "Executing all with Agglomerative and merge and dumping\n"

python main.py -g $GRAPH -a NormEig -c Agglomerative -k $KMAX -m -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormLap -c Agglomerative -k $KMAX -m -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormEigCol -c Agglomerative -k $KMAX -m -f $GRAPH.log --dump

now=$(date +"%T")
echo $now
echo "Executing all with Agglomerative and dumping\n"

python main.py -g $GRAPH -a NormEig -c Agglomerative -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormLap -c Agglomerative -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormEigCol -c Agglomerative -f $GRAPH.log --dump

now=$(date +"%T")
echo $now
echo "Executing all with K means modified and merge and dumping\n"

python main.py -g $GRAPH -a NormEig -c Kmeans_modified -k $KMAX -m -n $N -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormLap -c Kmeans_modified -k $KMAX -m -n $N -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormEigCol -c Kmeans_modified -k $KMAX -m -n $N -f $GRAPH.log --dump

now=$(date +"%T")
echo $now
echo "Executing all with K means modified and dumping\n"

python main.py -g $GRAPH -a NormEig -c Kmeans_modified -n $N -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormLap -c Kmeans_modified -n $N -f $GRAPH.log --dump
python main.py -g $GRAPH -a NormEigCol -c Kmeans_modified -n $N -f $GRAPH.log --dump

now=$(date +"%T")
echo $now
echo "Executing unormalized with merge\n"

python main.py -g $GRAPH -a Unorm -c Kmeans -k $KMAX -m -f $GRAPH.log
python main.py -g $GRAPH -a Unorm -c Agglomerative -k $KMAX -m -f $GRAPH.log
python main.py -g $GRAPH -a Unorm -c Gmm -k $KMAX -m -f $GRAPH.log
python main.py -g $GRAPH -a Unorm -c Kmeans_modified -k $KMAX -m -n $N -f $GRAPH.log

now=$(date +"%T")
echo $now
echo "Executing unormalized with K means\n"

python main.py -g $GRAPH -a Unorm -c Kmeans -f $GRAPH.log
python main.py -g $GRAPH -a Unorm -c Agglomerative -f $GRAPH.log
python main.py -g $GRAPH -a Unorm -c Gmm -f $GRAPH.log
python main.py -g $GRAPH -a Unorm -c Kmeans_modified -n $N -f $GRAPH.log

now=$(date +"%T")
echo $now
echo "Executing unormalized with merge and dumping\n"

python main.py -g $GRAPH -a Unorm -c Kmeans -k $KMAX -m -f $GRAPH.log --dump
python main.py -g $GRAPH -a Unorm -c Agglomerative -k $KMAX -m -f $GRAPH.log --dump
python main.py -g $GRAPH -a Unorm -c Gmm -k $KMAX -m -f $GRAPH.log --dump
python main.py -g $GRAPH -a Unorm -c Kmeans_modified -k $KMAX -m -n $N -f $GRAPH.log --dump

now=$(date +"%T")
echo $now
echo "Executing unormalized with K means and dumping\n"

python main.py -g $GRAPH -a Unorm -c Kmeans -f $GRAPH.log --dump
python main.py -g $GRAPH -a Unorm -c Agglomerative -f $GRAPH.log --dump
python main.py -g $GRAPH -a Unorm -c Gmm -f $GRAPH.log --dump
python main.py -g $GRAPH -a Unorm -c Kmeans_modified -n $N -f $GRAPH.log --dump


