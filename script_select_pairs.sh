source "${ABS_PATH}/handle_errors"
source "${ABS_PATH}/colors"

#=== FUNCTION =============================================
#
#
# Parameters:
#    $1 ->
#    $2 ->
#    $3 ->
#==========================================================
function run_parallel {
    echo -e "$RED>>FROM func$END" &> /dev/tty
    TOPIC=$1
    j=$2
    i=$3
    cp -r ../../SSARP_short SSARP_short.$j
    cp SSARP/run/alac_* SSARP_short.$j/run/
    cp SSARP/run/lac_train_TUBEfinal* SSARP_short.$j/run/


    name=`printf "data.$TOPIC/xxx%06d" $i`
    echo "cleaning 1 $name ---- thread " &> /dev/tty

    ../script_prune_pairs.sh $name $j $TOPIC 10 5 \
        &> /tmp/lixo.$j.$TOPIC

    cat /tmp/lixo.$j.$TOPIC

	pos=`grep "docs positivos coletados" /tmp/lixo.$j.$TOPIC | cut -d' ' -f4`
    neg=`grep "docs positivos coletados" /tmp/lixo.$j.$TOPIC | cut -d' ' -f11`

    real_posit=`cat $name \
        | cut -d' ' -f1 \
        | sort -k1 \
        | uniq \
        | join - goldendb \
        | wc -l`

    real_negat=`cat $name \
        | cut -d' ' -f1 \
        | sort -k1 \
        | uniq \
        | join - goldendb -v1 \
        |  wc -l`

    perda=$(($real_posit-$pos))
    perda_ac=$(($perda+$perda_ac))
    totalPos=$(($pos+$totalPos))

    echo "$name pos  $pos neg  $neg  real posit $real_posit real neg $real_negat perda $perda_ac totalPos $totalPos"

    cat x_posit_ssarp_end.$j x_negat_ssarp_end.$j \
        | cut -d' ' -f2  >> out_after_ssarp.$TOPIC
}

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
export TOPIC=$1
export file=$2              # JUDGECLASS
export Rel=$3
export rules=$4
export sliding_windows=$5

# echo "t $TOPIC j $file r $Rel rule $rules wind $sliding_windows"

echo "starting script stopping point with sliding windows of $5 ...."

# spliting dataset into files of 30 docs
sort -k1 rel.$TOPIC.fil > temp
mv temp rel.$TOPIC.fil

#-------------------------------------------------------------------------
# splits the final ranking into files with 30 lines each
#-------------------------------------------------------------------------
rm -r data.$TOPIC
mkdir data.$TOPIC
pushd data.$TOPIC

split -l 30 -d  ../result_ranking.$TOPIC xxx --suffix-length=6
len=`ls | wc -l`   # number of created files
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

#-------------------------------------------------------------------------
# se o numero de documentos é 0 termina
#-------------------------------------------------------------------------
if [ $len -le 0 ]; then
    cat x_posit_ssarp_end.* x_negat_ssarp_end.* | cut -d' ' -f2  >> out_after_ssarp.$TOPIC
    cat x_posit.* x_negat.* | cut -d' ' -f2  >> out_after_ssarp.$TOPIC
    cat out_after_ssarp.$TOPIC | sort | uniq > temp
    cp temp out_after_ssarp.$TOPIC

    echo "positivos `cat out_after_ssarp.$TOPIC | cut -d' ' -f1 | sort -k1 | uniq | join - rel.$TOPIC.fil |  wc -l` total `wc -l < out_after_ssarp.$TOPIC` input $totalPairsInput perda $perda_ac "
    exit

fi

cat x_posit_ssarp_end.* x_negat_ssarp_end.* \
    | cut -d' ' -f2  >> out_after_ssarp.$TOPIC
cat x_posit.* x_negat.* \
    | cut -d' ' -f2  >> out_after_ssarp.$TOPIC

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
step=-1

e_info "mid: $mid step: $step len: $len"

for i in $(seq $mid $step 0 ); do

    e_error "AAAAAAAAAAAAAA $i + $step + $step + A" &> /dev/tty

    if [ $(($i+$step+$step)) -ge $len ]; then
        step=$(($len-$i))
    fi


    l=0
    run_parallel $TOPIC $(($j+$l)) $(($l+$i))   &
    l=1
    run_parallel $TOPIC $(($j+$l)) $(($l+$i))   &
    l=2
    run_parallel $TOPIC $(($j+$l)) $(($l+$i))   &
    l=3
    run_parallel $TOPIC $(($j+$l)) $(($l+$i))   &
    l=4
    run_parallel $TOPIC $(($j+$l)) $(($l+$i))   &


    j=$(($j+$step))


    wait


    for l in $(seq 0 $(($step-1))); do
        temp=$(($l+$j-$step))
        pos=`grep "docs positivos coletados" /tmp/lixo.$temp.$TOPIC | cut -d' ' -f4`
        neg=`grep "docs positivos coletados" /tmp/lixo.$temp.$TOPIC | cut -d' ' -f11`
        printf " \n \n 00000000   $pos $neg  $flag---- $temp  $(($l+$i))"


        r=`cat out_after_ssarp.$TOPIC | cut -d' ' -f1 | sort -k1 |  uniq | join - goldendb |  wc -l`
        finalpares=`wc -l < out_after_ssarp.$TOPIC`
        total=$Rel
        recall=`echo "scale=6; ($r / $total)" | bc`
        precisao=`echo "scale=6; ($r / $finalpares)" | bc`



        cat SSARP_short.$temp/run/alac_round_lac_train_TUBEfinal.txt.$TOPIC >> SSARP/run/alac_lac_train_TUBEfinal.txt.$TOPIC
        cat SSARP_short.$temp/run/alac_round_lac_train_TUBEfinal.txt.$TOPIC >> SSARP/run/alac_full_lac_train_TUBEfinal.txt.$TOPIC
        printf "\n\n tamanho do treino `wc -l < SSARP/run/alac_full_lac_train_TUBEfinal.txt.$TOPIC`"

        if [ $neg -le 1 ]; then
            flag=$(($flag+1))

        else
            flag=0
        fi

        if [ $flag -eq 5 ]; then
            echo "------------------------------ breaking ..."
            exit
        fi
    done
done

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

echo "final REVEAL is $TOPIC $r ------$finalpares  Recall $recall  Precisao $precisao  posit  $r  labellingEffort `wc -l < training_set.$TOPIC`  onlyssarp  `wc -l < ssarp_labelling_full.$TOPIC` perda_ac $perda_ac sliding_windows $sliding_windows"
