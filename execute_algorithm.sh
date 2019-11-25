#!/bin/sh

ALGORITHM=$1

echo "Started execution of graph ca-GrQc with algorithm $ALGORITHM\n"
python main.py -g ca-GrQc -a $ALGORITHM
echo "\nFinished execution of graph ca-GrQc with algorithm $ALGORITHM\n\n"
echo "Started execution of graph ca-HepTh with algorithm $ALGORITHM\n"
python main.py -g ca-HepTh -a $ALGORITHM
echo "\nFinished execution of graph ca-HepTh with algorithm $ALGORITHM\n\n"
echo "Started execution of graph Oregon-1 with algorithm $ALGORITHM\n"
python main.py -g Oregon-1 -a $ALGORITHM
echo "\nFinished execution of graph Oregon-1 with algorithm $ALGORITHM\n\n"
echo "Started execution of graph ca-HepPh with algorithm $ALGORITHM\n"
python main.py -g ca-HepPh -a $ALGORITHM
echo "\nFinished execution of graph ca-HepPh with algorithm $ALGORITHM\n\n"
echo "Started execution of graph ca-AstroPh with algorithm $ALGORITHM\n"
python main.py -g ca-AstroPh -a $ALGORITHM
echo "\nFinished execution of graph ca-AstroPh with algorithm $ALGORITHM\n\n"
echo "Started execution of graph ca-CondMat with algorithm $ALGORITHM\n"
python main.py -g ca-CondMat -a $ALGORITHM
echo "\nFinished execution of graph ca-CondMat with algorithm $ALGORITHM\n\n"
echo "Started execution of graph soc-Epinions1 with algorithm $ALGORITHM\n"
python main.py -g soc-Epinions1 -a $ALGORITHM
echo "\nFinished execution of graph soc-Epinions1 with algorithm $ALGORITHM\n\n"
echo "Started execution of graph web-NotreDame with algorithm $ALGORITHM\n"
python main.py -g web-NotreDame -a $ALGORITHM
echo "\nFinished execution of graph web-NotreDame with algorithm $ALGORITHM\n\n"
echo "Started execution of graph roadNet-CA with algorithm $ALGORITHM\n"
python main.py -g roadNet-CA -a $ALGORITHM
echo "\nFinished execution of graph roadNet-CA with algorithm $ALGORITHM\n\n"