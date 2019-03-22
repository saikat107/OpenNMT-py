#!/usr/bin/env bash

for d_type in all filtered; do
    for type in concrete_unique abstract_unique; do
        for kind in token rule augmented.token; do
            echo "Training On Codit "$d_type" "$type" "$kind
            echo "============================================================================"
            echo ""
            python train.py -data c_data/processed/$d_type/$type.$kind -save_model c_models/$d_type.$type.$kind -gpuid 0 --type token
        done
    done
done
