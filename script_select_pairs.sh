  
  export TOPIC=$1
  export file=$2
  export Rel=$3
  export rules=$4
  export sliding_windows=$5
  source "${ABS_PATH}/handle_errors"
  echo "starting script stopping point with sliding windows of $5 ...."
  
  #spliting dataset into files of 30 docs
 # try
 # (
    sort -k1 rel.$TOPIC.fil > temp
    mv temp rel.$TOPIC.fil
    rm -r data.$TOPIC
    mkdir data.$TOPIC
    pushd data.$TOPIC
            
    split -l 30 -d  ../result_ranking.$TOPIC xxx --suffix-length=6
    len=`ls | wc -l`
    len=$(($len))
    popd
        
    mkdir data_plus.$TOPIC
    pushd data_plus.$TOPIC
    split -l 30 -d  ../result_plus.$TOPIC xxx --suffix-length=6
    len_plus=`ls | wc -l`
    len_plus=$(($len_plus-1))
    popd
    #     
        mem=0
        flag=0
        totalPos=0
        totalNeg=0
        totalPairsInput=0
        mem_i=0
    
        #cp seed_out.80.$TOPIC.arff seed_out
        exitstatus=0
        j=100
        rm out_after_ssarp.$TOPIC
        run=0
        run_memory=0
        run_attempy=0
        echo "len size $len"
        cat x_posit.* | sort | uniq >> posit_file
        sort -k2 posit_file > temp
        mv temp posit_file
        lo=1
        hi=$len 
        inicialize_allac=2
        count=0
        perda_ac=0
  #  ) 2> $STD_ERROR_OUT

   # catch || {
    #    kill -SIGUSR1 `ps --pid $$ -oppid=`; exit_on_error
    #}

    #se o numero de documentos é 0 termina 
    if [ $len -le 0 ]; then
      #  try
       # (
            cat x_posit_ssarp_end.* x_negat_ssarp_end.* | cut -d' ' -f2  >> out_after_ssarp.$TOPIC    
            cat x_posit.* x_negat.*   | cut -d' ' -f2  >> out_after_ssarp.$TOPIC
            cat out_after_ssarp.$TOPIC | sort | uniq > temp
            cp temp out_after_ssarp.$TOPIC    
            echo "positivos `cat out_after_ssarp.$TOPIC | cut -d' ' -f1 | sort -k1 |  uniq | join - rel.$TOPIC.fil |  wc -l` total `wc -l < out_after_ssarp.$TOPIC` input $totalPairsInput perda $perda_ac "
            exit
        #) 2> $STD_ERROR_OUT

        #catch || {
        #    kill -SIGUSR1 `ps --pid $$ -oppid=`; exit_on_error
        #}
    
    fi
    

    #try
    #(
        cat x_posit_ssarp_end.* x_negat_ssarp_end.* | cut -d' ' -f2  >> out_after_ssarp.$TOPIC   
        cat x_posit.* x_negat.*   | cut -d' ' -f2  >> out_after_ssarp.$TOPIC
    
    
    
        cat out_after_ssarp.$TOPIC | sort | uniq > temp
        cp temp out_after_ssarp.$TOPIC
    
        r=`cat out_after_ssarp.$TOPIC | cut -d' ' -f1 | sort -k1 |  uniq | join - rel.$TOPIC.fil |  wc -l`
        finalpares=`wc -l < out_after_ssarp.$TOPIC`
        total=$Rel
        recall=`echo "scale=6; ($r / $total)" | bc`
        precisao=`echo "scale=6; ($r / $finalpares)" | bc`
    
        
        
        echo "cleaning from $len until 0 stopping when $pos equal zero "
        flag=0
        temp=0
        mid=$len
        new_mid=0
        mid=$(($mid-1)) #start from zero 
    #) 2> $STD_ERROR_OUT

    #catch || {
    #    kill -SIGUSR1 `ps --pid $$ -oppid=`; exit_on_error
    #}
    for i in $(seq $mid -1 0 ); do    
    #	try
    #    (
            j=$(($j+1))
            temp=$(($temp+1))

            name=`printf "data.$TOPIC/xxx%06d" $i` #create name of datafile
            echo "cleaning 2 $name ---- $i"

            cat $name | sort > temp
            cat x_negat.* x_posit.* | cut -d' ' -f2 | sort | join - temp -v2 | sort > clean_data #data without already labeled docs 
        echo "script prune pairs..."
            .././script_prune_pairs.sh clean_data  $j $TOPIC $inicialize_allac $rules $sliding_windows >  /tmp/lixo.$TOPIC
        inicialize_allac=3
            trap "echo EXIT ERROR.>&2;exit" SIGUSR2
            cat /tmp/lixo.$TOPIC
            
            ####################################
            #read -n1 ans
            #####################################
            
            pos=`grep "docs positivos coletados" /tmp/lixo.$TOPIC | cut -d' ' -f4`
            neg=`grep "docs negativos coletados" /tmp/lixo.$TOPIC | cut -d' ' -f11`      
            real_posit=`cat clean_data | cut -d' ' -f1 | sort -k1 |  uniq | join - rel.$TOPIC.fil |  wc -l`
            perda=$(($real_posit-$pos))
            perda_ac=$(($perda+$perda_ac))
            
            echo "$name pos  $pos neg  $neg  real posit $real_posit  perda $perda_ac"
            cat x_posit_ssarp_end.$j x_negat_ssarp_end.$j | cut -d' ' -f2  >> out_after_ssarp.$TOPIC
            
            r=`cat out_after_ssarp.$TOPIC | cut -d' ' -f1 | sort -k1 |  uniq | join - rel.$TOPIC.fil |  wc -l`
            finalpares=`wc -l < out_after_ssarp.$TOPIC`
            total=$Rel
            recall=`echo "scale=6; ($r / $total)" | bc`
            precisao=`echo "scale=6; ($r / $finalpares)" | bc`
            echo "****final result AM is $TOPIC $r ------$finalpares  Recall $recall  Precisao $precisao  posit  $pos $net labellingEffort `wc -l < out_after_ssarp.$TOPIC` perda_ac $perda_ac "
            echo "flag $flag"
     #   ) 2> $STD_ERROR_OUT

      #  catch || {
       #     kill -SIGUSR1 `ps --pid $$ -oppid=`; exit_on_error
        #}

        if [ $pos -le 0 ]; then
                echo "increase flag value"
                flag=$(($flag+1))
        else
                echo "new flag equal zero"
                flag=0
        fi

  	                
        if [ $flag -eq $sliding_windows ]; then
            new_mid=$i;
            echo "break because flag sliding windows"
            break;
        fi
        
    done
    flag=0
    
    temp=0
    if [ $new_mid -le 0 ]; then
        echo "######inside active plus $len_plus"
        for i in $(seq $len_plus -1 0 ); do
         #   try
          #  (
                name=`printf "data_plus.$TOPIC/xxx%06d" $i`
                echo "inside active plus $name ---- $i"
                j=$(($j+1))
                temp=$(($temp+1))
                cat $name | sort > temp
                cat x_negat.* x_posit.* | cut -d' ' -f2 | sort | join - temp -v2 | sort > new_data
                #echo "positivos no seed `cat $name | sort | join - posit_file -2 2  | wc -l `"
                .././script_prune_pairs.sh new_data  $j $TOPIC $inicialize_allac $rules $sliding_windows &> /tmp/lixo.$TOPIC  
                cat /tmp/lixo.$TOPIC  
                pos=`grep "docs positivos coletados" /tmp/lixo.$TOPIC | cut -d' ' -f4`
                neg=`grep "docs positivos coletados" /tmp/lixo.$TOPIC | cut -d' ' -f11`      
                real_posit=`cat new_data | cut -d' ' -f1 | sort -k1 |  uniq | join - rel.$TOPIC.fil |  wc -l`
                perda=$(($real_posit-$pos))
                perda_ac=$(($perda+$perda_ac))
                
                echo "$name pos  $pos neg  $neg  real posit $real_posit  perda $perda_ac"
                cat x_posit_ssarp_end.$j x_negat_ssarp_end.$j | cut -d' ' -f2  >> out_after_ssarp.$TOPIC
                
                r=`cat out_after_ssarp.$TOPIC | cut -d' ' -f1 | sort -k1 |  uniq | join - rel.$TOPIC.fil |  wc -l`
                finalpares=`wc -l < out_after_ssarp.$TOPIC`
                #total=`wc -l < rel.$TOPIC.fil`
                total=$Rel
                recall=`echo "scale=6; ($r / $total)" | bc`
                precisao=`echo "scale=6; ($r / $finalpares)" | bc`
                echo "****final result AM is $TOPIC $r ------$finalpares  Recall $recall  Precisao $precisao  posit  $pos $net labellingEffort `wc -l < out_after_ssarp.$TOPIC` perda_ac $perda_ac "
                echo "flag $flag"
           # ) 2> $STD_ERROR_OUT

            #catch || {
            #    kill -SIGUSR1 `ps --pid $$ -oppid=`; exit_on_error
            #}
		
           # if (( $(echo "$recall > 0.90" |bc -l) )); then
                if [ $pos -le 0 ]; then
                    flag=$(($flag+1))
                else
                    flag=0
                    
                fi
                if [ $flag -eq $sliding_windows ]; then
                    new_mid=$i;
                    break;
                fi
		if [ $j -ge 2000 ]; then
                        echo "##break because of stopping $i"
			break;
		fi            
        done
    fi
    #try
    #(
        cat x_posit_ssarp_end.* x_negat_ssarp_end.* | cut -d' ' -f2  > training_set.$TOPIC

    cat only_levels_ssarp.$TOPIC	    training_set.$TOPIC | sort | uniq  >  ssarp_labelling_full.$TOPIC


        cat x_posit.* x_negat.* | cut -d' ' -f2  >> training_set.$TOPIC
        sort training_set.$TOPIC | sort | uniq > temp
        mv temp training_set.$TOPIC    
        cat x_posit_ssarp_end.* x_negat_ssarp_end.* | cut -d' ' -f2  >> out_after_ssarp.$TOPIC


        cat out_after_ssarp.$TOPIC | sort | uniq > temp
        cp temp out_after_ssarp.$TOPIC
    
        r=`cat out_after_ssarp.$TOPIC | cut -d' ' -f1 | sort -k1 |  uniq | join - goldendb |  wc -l`
        finalpares=`wc -l < out_after_ssarp.$TOPIC`
        total=`wc -l < goldendb`
        echo "total de relevantes $total"
        recall=`echo "scale=6; ($r / $total)" | bc`
        precisao=`echo "scale=6; ($r / $finalpares)" | bc`
        #echo "positivos `cat out_after_ssarp.$TOPIC | cut -d' ' -f1 | sort -k1 |  uniq | join - goldendb |  wc -l` total `wc -l < out_after_ssarp.$TOPIC` input $totalPairsInput perda
        #$perda_ac "
        echo "positivos $r total $finalpares recal  $recall precision $precision perda $perda_ac"
    #) 2> $STD_ERROR_OUT

    #catch || {
    #    kill -SIGUSR1 `ps --pid $$ -oppid=`; exit_on_error
    #}
	
echo "final REVEAL is $TOPIC $r ------$finalpares  Recall $recall  Precisao $precisao  posit  $r  labellingEffort `wc -l < training_set.$TOPIC`  onlyssarp  `wc -l < ssarp_labelling_full.$TOPIC` perda_ac $perda_ac sliding_windows $sliding_windows"
