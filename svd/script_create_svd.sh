
input=$1
cp $input ../svd/athome1
cat $2 >> ../svd/athome1
pushd ../svd


python3 trans.py <(sed  's/^/0 /' <(cut -d' ' -f2- athome1))  out

sed -i 's/^0//' out

paste <(cut -d' ' -f1 athome1) out -d ' ' | sed 's/  / /'  > $input.svd
popd

mv ../svd/$input.svd .
echo "file $input.svd was created"




