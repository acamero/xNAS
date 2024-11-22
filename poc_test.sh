#!/bin/bash

for i in $(seq 11 30);
do
    echo "python nsga2_pareto_ens.py --seed $i --epochs 10 --valsplit 0.2 --fitsplit 1000 --popsize 12 --generations 100 --oa"
    python nsga2_pareto_ens.py --seed $i --epochs 10 --valsplit 0.2 --fitsplit 1000 --popsize 12 --generations 100 --oa
done

for i in $(seq 11 30);
do
    echo "python nsga2_pareto_ens.py --seed $i --epochs 10 --valsplit 0.2 --fitsplit 1000 --popsize 12 --generations 100"
    python nsga2_pareto_ens.py --seed $i --epochs 10 --valsplit 0.2 --fitsplit 1000 --popsize 12 --generations 100
done

