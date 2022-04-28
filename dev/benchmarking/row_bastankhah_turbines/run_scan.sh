n_s=$1
scd=$2
t0=$3
t1=$4
tstep=$5
shift 5
opt="$@"

echo "n_s = $n_s, scheduler = $scd"
echo "t0 = $t0, t1 = $t1, tstep = $tstep"
echo -e "opt = $opt \n"

for (( t=$t0; t<=$t1; t+=$tstep ))
do
    python run.py $n_s $t $scd $opt
done


