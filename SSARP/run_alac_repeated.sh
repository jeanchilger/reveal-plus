#!/bin/bash

source "${ABS_PATH}/handle_errors"
source "${ABS_PATH}/colors"

loop=1
count=0;
rule=$3
if [ ! -f alac_$1 ]; then touch alac_$1; fi
if [ ! -d result_temp_$1 ]; then mkdir result_temp_$1; fi


if [ ! -f alac_full_$1 ]; then touch alac_full_$1; fi

echo "starting alac test threshold  ../../alac`echo $rule`  $1 $2  with size `wc -l < $1`"

while [ $loop == 1 ]; do
    ../alac`echo $rule` -i alac_$1 -t $1 -s 1 -m 3 -e 1000000000 -c 0.001 -j 1 -d 3 -o 1 > result_temp_$1/result_temp$count.txt

    # try
    # (
        instance=`cat result_temp_$1/result_temp$count.txt | grep inserting | awk '{ print $3 }'`;
        newclass=`cat result_temp_$1/result_temp$count.txt | grep "New CLASS" | awk '{ print $4 }'`;
        #   echo "../../alac -i alac_$1 -t $1 -s 1 -m 3 -e 1000000000 -c 0.001 -j 1 -d 3 -o 1 > result_temp_$1/result_temp$count.txt"
        exists=`cat alac_$1 | grep "^$instance\ "`;

    # ) 2> $STD_ERROR_OUT
    #
    # catch || {
    #     exit_on_error
    # }

    if [ "$exists" == "" ]; then
        echo inserting instance $instance into alac_$1 `cat $1 | grep "^$instance\ " | cut -d' ' -f2`

        if [ "$newclass" != "" ]; then
          echo class changed to $newclass;
        fi

        #`wc -l < docfils_temp.$TOPIC`
        #echo `cat $1 | grep "$instance\" | grep CLASS=0`
        cat $1 | grep "^$instance\ " |  grep "CLASS=0" >> alac_$1;
        # cat $1 | grep "^$instance\ " |  grep "CLASS=0">> alac_$1;
        #| grep "CLASS=0"
        cat $1 | grep "^$instance\ " >> alac_full_$1;
        cp -f $1 temp_$1;
        cat $1 | grep "^$instance\ " | grep "CLASS=1" | awk '{ print $1 }' |  while read instance; do  sed -i  "/^$instance /d" temp_$1 ; break;  done
        cp -f temp_$1 $1;
        count=$(($count+1));
        size=`wc -l < $1`;

        if [ $size -le 1 ]; then
            cat $1 >> alac_full_$1;
            echo "stoping ssar selection because file is empty"
            break;
        fi

    else
        echo instance $instance already inserted... terminating;
        echo total of $count instances inserted into training set;
        cat $1 | grep "^$instance\ " >> instanceFile
        sort instanceFile | uniq > temp
        mv temp instanceFile
        #echo `cat $1 |grep "^$instance\ "`
        loop=0;

    fi
done
