#!/usr/bin/env bash
#if [[ ! -d c_data/processed ]]; then
#    mkdir c_data/processed
#fi
#
#if [[ ! -d c_data/processed/all ]]; then
#    mkdir c_data/processed/all
#fi

#if [[ ! -d c_data/processed/filtered ]]; then
#    mkdir c_data/processed/filtered
#fi

for fil in all; do
    for type in concrete;
    do
        for kind in rule token augmented.token augmented.rule;do
            echo $type' '$kind
            python preprocess.py -train_src c_data/raw/$fil/$type/train/prev.$kind \
                  -train_tgt c_data/raw/$fil/$type/train/next.$kind \
                  -valid_src c_data/raw/$fil/$type/valid/prev.$kind \
                  -valid_tgt c_data/raw/$fil/$type/valid/next.$kind \
                  -save_data c_data/processed/$fil/$type'.'$kind \
                  -share_vocab;
        done
    done
done
