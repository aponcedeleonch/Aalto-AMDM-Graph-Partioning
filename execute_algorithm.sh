#!/bin/sh

ALGORITHM=$1
CLUSTERING=$2

echo "Started execution of graph ca-GrQc with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g ca-GrQc -a $ALGORITHM -c $CLUSTERING --dump
echo "\nFinished execution of graph ca-GrQc with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph ca-HepTh with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g ca-HepTh -a $ALGORITHM -c $CLUSTERING --dump
echo "\nFinished execution of graph ca-HepTh with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph Oregon-1 with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g Oregon-1 -a $ALGORITHM -c $CLUSTERING --dump
echo "\nFinished execution of graph Oregon-1 with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph ca-HepPh with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g ca-HepPh -a $ALGORITHM -c $CLUSTERING --dump
echo "\nFinished execution of graph ca-HepPh with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph ca-AstroPh with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g ca-AstroPh -a $ALGORITHM -c $CLUSTERING --dump
echo "\nFinished execution of graph ca-AstroPh with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph ca-CondMat with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g ca-CondMat -a $ALGORITHM -c $CLUSTERING --dump
echo "\nFinished execution of graph ca-CondMat with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph soc-Epinions1 with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g soc-Epinions1 -a $ALGORITHM -c $CLUSTERING --dump
echo "\nFinished execution of graph soc-Epinions1 with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph web-NotreDame with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g web-NotreDame -a $ALGORITHM -c $CLUSTERING --dump
echo "\nFinished execution of graph web-NotreDame with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph roadNet-CA with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g roadNet-CA -a $ALGORITHM -c $CLUSTERING --dump
echo "\nFinished execution of graph roadNet-CA with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"


echo "Started execution of graph ca-GrQc with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g ca-GrQc -a $ALGORITHM -c $CLUSTERING
echo "\nFinished execution of graph ca-GrQc with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph ca-HepTh with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g ca-HepTh -a $ALGORITHM -c $CLUSTERING
echo "\nFinished execution of graph ca-HepTh with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph Oregon-1 with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g Oregon-1 -a $ALGORITHM -c $CLUSTERING
echo "\nFinished execution of graph Oregon-1 with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph ca-HepPh with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g ca-HepPh -a $ALGORITHM -c $CLUSTERING
echo "\nFinished execution of graph ca-HepPh with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph ca-AstroPh with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g ca-AstroPh -a $ALGORITHM -c $CLUSTERING
echo "\nFinished execution of graph ca-AstroPh with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph ca-CondMat with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g ca-CondMat -a $ALGORITHM -c $CLUSTERING
echo "\nFinished execution of graph ca-CondMat with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph soc-Epinions1 with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g soc-Epinions1 -a $ALGORITHM -c $CLUSTERING
echo "\nFinished execution of graph soc-Epinions1 with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph web-NotreDame with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g web-NotreDame -a $ALGORITHM -c $CLUSTERING
echo "\nFinished execution of graph web-NotreDame with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
echo "Started execution of graph roadNet-CA with algorithm $ALGORITHM and clustering $CLUSTERING\n"
python main.py -g roadNet-CA -a $ALGORITHM -c $CLUSTERING
echo "\nFinished execution of graph roadNet-CA with algorithm $ALGORITHM and clustering $CLUSTERING\n\n"
