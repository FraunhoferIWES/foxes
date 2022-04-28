n_s=100000
cpus=10
thrds=2
scd="distributed"
cs="100 500 1000 2000 5000 10000"
ts="10 100 250 500"
w="Bastankhah_linear"
r="centre"
l="foxes0_10"
o="results_Bastankhah_linear_${r}.csv"

opt="$@"
echo -e "opt = $opt \n"

for c in $cs
do
    for t in $ts
    do
        echo -e "RUNNING $l: c = $c, t = $t, o = $o"
        echo python run.py $n_s $t $scd -c $c --n_cpus $cpus --threads_per_cpu $thrds -l $l -w $w -r $r -o $o $opt
        echo -e "\n"

        python run.py $n_s $t $scd -c $c --n_cpus $cpus --threads_per_cpu $thrds -l $l -w $w -r $r -o $o $opt
    done
done


