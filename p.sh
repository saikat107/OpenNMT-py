#!/usr/bin/env bash
if [[ ! -d rule_based_data/processed ]]; then
    mkdir rule_based_data/processed
fi

if [[ ! -d rule_based_data/processed/all ]]; then
    mkdir rule_based_data/processed/all
fi

if [[ ! -d rule_based_data/processed/filtered ]]; then
    mkdir rule_based_data/processed/filtered
fi

for fil in all filtered; do
    for type in original;
    do
        for size in small;do
            for kind in rule token augmented.token augmented.rule;do
                echo $type' '$size' '$kind
                python preprocess.py -train_src rule_based_data/raw/$fil/$type'_'$size/train/prev.$kind -train_tgt rule_based_data/raw/$fil/$type'_'$size/train/next.$kind -valid_src rule_based_data/raw/$fil/$type'_'$size/valid/prev.$kind -valid_tgt rule_based_data/raw/$fil/$type'_'$size/valid/next.$kind -save_data rule_based_data/processed/$fil/$type'.'$size'.'$kind -share_vocab
            done
        done
    done
done
