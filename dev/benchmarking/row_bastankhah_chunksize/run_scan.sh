n_s=$1
n_t=$2
scd=$3
c0=$4
c1=$5
cstep=$6
shift 6
opt="$@"

echo "n_s = $n_s, n_t = $n_t, scheduler = $scd"
echo "c0=$c0, c1=$c1, cstep = $cstep"
echo -e "opt = $opt \n"

for (( c=$c0; c<=$c1; c+=$cstep ))
do
    python run.py $n_s $n_t $scd -c $c $opt
done


