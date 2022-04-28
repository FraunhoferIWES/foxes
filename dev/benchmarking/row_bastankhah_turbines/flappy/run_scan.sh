n_s=$1
t0=$2
t1=$3
tstep=$4
shift 4
opt="$@"

echo "n_s = $n_s"
echo "t0=$t0, t1=$t1, tstep = $tstep"
echo -e "opt = $opt \n"

for (( t=$t0; t<=$t1; t+=$tstep ))
do
    python run.py $n_s $t $opt
done


