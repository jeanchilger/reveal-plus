#!/bin/bash

suffix=$4

if [ "$suffix" == "" ]; then suffix="B10"; fi

export CLASSPATH=../../weka.jar

# Create ChiSquared Attributes

if [ -f $2 ]; then rm -f $2; fi

i=1 
while [ $i -le $(($3)) ]; do 

  echo -n " $i"
  if [ -f train-$suffix-$i-0LL.hist ]; then 
    #echo "java weka.attributeSelection.ChiSquaredAttributeEval -M  -s "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1" -i $1 -c $i"
    java weka.attributeSelection.ChiSquaredAttributeEval -M  -s "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1" -i $1 -c $i | grep Selected | awk '{ print $3 }' >> $2
  fi 
  i=$(($i+1))

done 

#java weka.attributeSelection.ChiSquaredAttributeEval -M  -s "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1" -i $1 | grep Selected | awk '{ print $3 }' > $2

