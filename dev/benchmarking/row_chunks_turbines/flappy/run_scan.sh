n_s=100000
cpus=10
cs="100 500 1000 2000 5000 10000"
ts="10 100 250 500"
w="Bastankhah_rotor"
s="wind_linear"
r="centre"
l="flappy_10"
o="../results_Bastankhah_linear_${r}.csv"

opt="$@"
echo -e "opt = $opt \n"

for c in $cs
do
    for t in $ts
    do
        echo -e "RUNNING $l: c = $c, t = $t"
        echo python run.py $n_s $t -c $c --n_cpus $cpus -l $l -w $w -s $s -r $r -o $o $opt
        echo -e "\n"
        
        python run.py $n_s $t -c $c --n_cpus $cpus -l $l -w $w -s $s -r $r -o $o $opt
    done
done


