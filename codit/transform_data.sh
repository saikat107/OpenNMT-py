#!/usr/bin/env bash

cwd=`pwd`
export PYTHONPATH=$PYTHONPATH:cwd

###### Create Necessary Directories ###########
if [[ ! -d rule_based_data ]]; then
    mkdir rule_based_data
fi

if [[ ! -d rule_based_data/raw ]]; then
    mkdir rule_based_data/raw
fi

if [[ ! -d rule_based_data/raw/all ]]; then
    mkdir rule_based_data/raw/all
fi

if [[ ! -d rule_based_data/raw/filtered ]]; then
    mkdir rule_based_data/raw/filtered
fi


for type in original concrete;
do
        for size in small medium;
        do
                echo $type
                echo $size
                python codit/create_transformation_data.py -data /local/saikatc/Research/icse_data_$type\
                                -source all -name $size -valid eval -output rule_based_data/raw/all/$type'_'$size
        done
done

for type in original concrete;
do
        for size in small medium;
        do
                echo $type
                echo $size
                python codit/create_transformation_data.py -data /local/saikatc/Research/icse_data_$type\
                -source all -name $size -valid eval -output rule_based_data/raw/filtered/$type'_'$size -exclude_no_structure_change
        done
done
