    
TOPIC=$3
N=$2
flag=$4
rules=$5

echo "starting run ssarpx\n"
#identify docs pos and neg
cat $1 | sort | uniq | join - rel.$TOPIC.fil | cut -d' ' -f1 | sed -e 's/^/1 /' > temp_posit.$N.$TOPIC
cat $1 | sort | uniq | join - rel.$TOPIC.fil -v1 | cut -d' ' -f1 | sed -e 's/^/-1 /' > temp_negat.$N.$TOPIC

#produce the training set file used by SSARP
cat temp_posit.$N.$TOPIC temp_negat.$N.$TOPIC |  sort -k2  | join - ../"$file".svm.fil.svd  -2 1 -1 2 > trainset.$N.$TOPIC
cut -d ' ' -f2- trainset.$N.$TOPIC  > trainsetB.$N.$TOPIC
python3 ../svd/convert_txt.py trainsetB.$N.$TOPIC trainset.$N.$TOPIC.arff rel.$TOPIC.fil

cp seed_out.10.* seed_out
#clean some files 
cp trainset.$N.$TOPIC ../SSARP/run/
cp seed_out ../SSARP/run/
cp trainset.$N.$TOPIC.arff ../SSARP/run/
cd ../SSARP/run/ 

#run active learning
./SSARPX.sh trainset.$N.$TOPIC trainset.$N.$TOPIC 50 $N seed_out $TOPIC $flag $rules &>   log_ssarp_stoping_$N

#store result 
cat label.$N.$TOPIC > /tmp/ssarp$N.$TOPIC
cat log_ssarp_stoping_$N
cd -
mv /tmp/ssarp$N.$TOPIC .    

   

#compute docs positivos and negativos          
cat ssarp$N.$TOPIC   | sort -k 2 | uniq | join - rel.$TOPIC.fil  | cut -d' ' -f1 | sed -e 's/^/1 /' > x_posit_ssarp_end.$N
cat ssarp$N.$TOPIC   | sort -k 2 | uniq | join - rel.$TOPIC.fil -v1 | cut -d' ' -f1 | sed -e 's/^/-1 /' > x_negat_ssarp_end.$N


cat x_posit_ssarp_end.$N x_negat_ssarp_end.$N | sort | uniq > ssarpout$N.$TOPIC
cut -d' ' -f2  ssarpout$N.$TOPIC > sub_new_ssarp.$N.$TOPIC

echo "docs positivos coletados `wc -l < x_posit_ssarp_end.$N`   docs negativos ----  `wc -l < x_negat_ssarp_end.$N`"



