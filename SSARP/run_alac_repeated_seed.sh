#!/bin/bash

loop=1
count=0;

if [ ! -f alac_$1 ]; then touch alac_$1; fi
if [ ! -d result_temp_$1 ]; then mkdir result_temp_$1; fi


if [ ! -f alac_full_$1 ]; then touch alac_full_$1; fi
#echo alac_$1
# confiança 0,001
#e= cache size
#m=MAX_RULE_SIZE
#cconfiança
#j=MAX_JUDGEMENTS
#d
while [ $loop == 1 ]; do
  ../../alac -i alac_$1 -t $1 -s 1 -m 3 -e 1000000000 -c 0.001 -j 1 -d 3 -o 1 > result_temp_$1/result_temp$count.txt
  instance=`cat result_temp_$1/result_temp$count.txt | grep inserting | awk '{ print $3 }'`; 
  newclass=`cat result_temp_$1/result_temp$count.txt | grep "New CLASS" | awk '{ print $4 }'`;
#   echo "../../alac -i alac_$1 -t $1 -s 1 -m 3 -e 1000000000 -c 0.001 -j 1 -d 3 -o 1 > result_temp_$1/result_temp$count.txt"
  exists=`cat alac_$1 | grep "^$instance\ "`; 
  if [ "$exists" == "" ]; then 
    echo inserting instance $instance into alac_$1
    if [ "$newclass" != "" ]; then
      echo class changed to $newclass;
    fi
    #`wc -l < docfils_temp.$TOPIC`
    #echo `cat $1 | grep "$instance\" | grep CLASS=0`
    cat $1 | grep "^$instance\ ">> alac_$1;
   # cat $1 | grep "^$instance\ " |  grep "CLASS=0">> alac_$1; 
    #| grep "CLASS=0"
    cat $1 | grep "^$instance\ " >> alac_full_$1;
#     cp $1 temp;
#     cat $1 | grep "^$instance\ " | grep "CLASS=1" | awk '{ print $1 }' |  while read instance; do  sed -i  "/^$instance /d" temp ;  done
#     cp temp $1;
    count=$(($count+1));
    size=`wc -l < $1`;
    if [ $count -gt 100 ]; then
        
        echo "selecionou 30 pares"
        break;
    fi
  else
    echo instance $instance already inserted... terminating;
    echo total of $count instances inserted into training set;
    loop=0;

  fi
done


