
#!/bin/bash

loop=1

count=0;
rule=$3
max_rule=$4
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

echo "starting alac test threshold  ../../alac`echo $rule`  $1 $2  with size `wc -l < $1` with $max_rule rules"
while [ $loop == 1 ]; do
 # echo "../alac`echo $rule` -i alac_$1 -t $1 -s 1 -m $max_rule -e 1000000000 -c 0.001 -j 1 -d 3 -o 1 > result_temp_$1/result_temp$count.txt"
  echo "run alac "
  ../alac`echo $rule` -i alac_$1 -t $1 -s 1 -m $max_rule -e 1000000000 -c 0.001 -j 1 -d 3 -o 1 > result_temp_$1/result_temp$count.txt
  #cat result_temp_$1/result_temp$count.txt
  #echo "---------------------"
  time=0
  intern=1
  
  
  
  while [ $intern == 1 ]; do
  instance=`cat result_temp_$1/result_temp$count.txt | grep "^pos $time " | awk '{ print $3 }'`; 
  echo "------$instance ----- pos $time"
  if [ "$instance" == "" ]; then 
    loop=0
    break;
   fi
#   echo "../../alac -i alac_$1 -t $1 -s 1 -m 3 -e 1000000000 -c 0.001 -j 1 -d 3 -o 1 > result_temp_$1/result_temp$count.txt"
  exists=`cat alac_$1 | grep "^$instance\ "`; 
  if [ "$exists" == "" ]; then 
    echo inserting instance $instance into alac_$1 `cat $1 | grep "^$instance\ " | cut -d' ' -f2 `  
    
   # echo "with size `wc -l < $1` name $1"
    
    cat $1 | grep "^$instance\ " |  grep "CLASS=0" >> alac_$1;
   # cat $1 | grep "^$instance\ " |  grep "CLASS=0">> alac_$1; 
    #| grep "CLASS=0"
    cat $1 | grep "^$instance\ " >> alac_full_$1;
    #cp $1 temp_$1;
    
    #cat $1 | grep "^$instance\ " | grep "CLASS=1" | awk '{ print $1 }' |  while read instance; do  sed -i  "/^$instance /d" temp_$1 ; break;  done
    #cp temp_$1 $1;
   # read -n1
    #echo `cat $1 | grep "^$instance\ "`
    
    
    var=`grep -n "^$instance\ " $1  | grep  "CLASS=1" | cut -d: -f1`
   
    
    if [ "$var" == "" ]; then 
      #echo "xxx----"
      #echo value cat $1 | grep " $instance " 
     # echo "-----`cat $1 | grep "^$instance\ " `-----"
      echo "instance negative $instance"  
       
    else
        sed -i  "${var}d" $1    #> temp
        #mv temp $1
       # echo "-----removing line $var ----- `wc -l < $1`"
    fi
   
    
    
    label=`cat $1 | grep "^$instance\ " |  grep "CLASS=0"| wc -l`
    if [ $label -eq 1 ]; then 
        echo "class equal to zero so stop intern"
        intern=0
        break;
    fi

    size=`wc -l < $1`;
    if [ $size -le 1 ]; then
        cat $1 >> alac_full_$1;
        echo "stoping ssar selection because file is empty 1  $size"
        break;
    fi
    
  else
   
#     cat $1 | grep "^$instance\ " >> instanceFile
#     sort instanceFile | uniq > temp
#     mv temp instanceFile
    #echo `cat $1 |grep "^$instance\ "`
    if [ $time -eq 0 ]; then
        echo instance $instance already inserted... terminating;
        echo total of $count instances inserted into training set;
        intern=0
        loop=0;
        break;
    else
        echo "end intern selection"
        count=$(($count+1));
        intern=0
    fi
    
  fi 
  time=$((time+1))
  done
  
  size=`wc -l < $1`;
    if [ $size -le 1 ]; then
        cat $1 >> alac_full_$1;
        echo "---stoping ssar selection because file is empty 2  $size ---- "
        break;
    fi
done

echo "ENDDDDDDDDDDDDDD"

