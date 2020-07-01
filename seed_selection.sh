#!/bin/bash


source "${ABS_PATH}/handle_errors"
source "${ABS_PATH}/colors"

JUDGECLASS=$1; shift
VERBOSE=false; [[ $1 == "true" ]] && VERBOSE=true; shift
TOPIC=$1; shift


$VERBOSE && e_success "Starting."


export discretize=0
#loop over the ranking
constante=30
for i in `seq 0 10`; do
   step=$((i+1))
   begin=$((i*constante+1))
   end=$((i*constante+constante))
   sed -n "${begin},${end}p" SeedRanking`echo $TOPIC` > top_small
   #cat ranking/ranking`echo $TOPIC`.txt> top_small
   #head -n 10 goldendb.$TOPIC
   printf "\n\n $begin $end--------- "
   echo "Relevants in the seed `cat top_small | sort | join - goldendb.$TOPIC | wc -l `" 
   
   round=$i
   source ../active_learning $CORPUS `echo $VERBOSE` $TOPIC $round $discretize
   discretize=1
   
   #cat top_small
   pos=`wc -l < x_posit.$step`
   if [ $pos -ge 1 ]
   then 
       echo "we found a  pos doc"
       break;
   fi
done
  
  
  
