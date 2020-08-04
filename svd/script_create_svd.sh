
input=$1
temp=$4
cp $input ../svd/$temp
cat $2 >> ../svd/$temp
pushd ../svd


python3 trans.py <(sed  's/^/0 /' <(cut -d' ' -f2- $temp))  out

sed -i 's/^0//' out

paste <(cut -d' ' -f1 $temp) out -d ' ' | sed 's/  / /'  > $input.svd
popd

mv ../svd/$input.svd .
echo "file $input.svd was created"




