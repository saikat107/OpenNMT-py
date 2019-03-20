#!/usr/bin/env bash

for fil in br original; do
        for kind in rule token augmented.token;do
            echo $fil' '$kind
            python train.py -data defj_experiment/data/processed/$fil.$kind \
            -save_model defj_experiment/models/$fil.$kind -gpuid 0 --type token
        done
done