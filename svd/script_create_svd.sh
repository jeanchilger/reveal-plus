
input=$1
cp $input /tmp/athome1
cat $2 >> /tmp/athome1
pushd ../svd
cut -d' ' -f2- /tmp/athome1 > /tmp/athome1_2
cut -d' ' -f1 /tmp/athome1 > /tmp/labels
sed -i 's/^/0 /' /tmp/athome1_2

python3 trans.py /tmp/athome1_2  out 

sed -i 's/^0//' out
paste /tmp/labels out -d ' ' | sed 's/  / /'  > $input.svd
popd
echo "file $input.svd was created"
