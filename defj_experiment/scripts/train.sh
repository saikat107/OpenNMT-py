#!/usr/bin/env bash

#for fil in br original; do
#        for kind in rule token augmented.token;do
#            echo $fil' '$kind
#            python train.py -data defj_experiment/data/processed/$fil.$kind \
#            -save_model defj_experiment/models/$fil.$kind -gpuid 0 --type token
#        done
#done

for project in Chart Lang Math Time Closure Mockito;do
    for fil in br original; do
        for kind in token augmented.token;do
            echo $project' '$fil' '$kind
            python train.py -data defj_experiment/data/processed/$fil.$kind.$project \
            -save_model defj_experiment/models/$fil.$kind.$project -gpuid 0 --type token
        done
    done
done