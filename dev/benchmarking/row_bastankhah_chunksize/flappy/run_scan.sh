n_s=$1
n_t=$2
c0=$3
c1=$4
cstep=$5
shift 5
opt="$@"

echo "n_s = $n_s, n_t = $n_t"
echo "c0=$c0, c1=$c1, cstep = $cstep"
echo -e "opt = $opt \n"

for (( c=$c0; c<=$c1; c+=$cstep ))
do
    python run.py $n_s $n_t -c $c $opt
done


