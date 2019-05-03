#!/usr/bin/env bash
if [[ $# != 4 ]]; then
    echo "Must provide 4 arguments "
    echo "dataset<icse, codit>, ditasize<all, filtered>, datatype<concrete, abstract, concrete_small, abstract_small, concrete_unique, abstract_unique> <beam_size>"
    exit
fi

dataset=$1 # icse, codit
datasize=$2 # all, filtered, original
datatype=$3 # concrete, abstract, concrete_small, abstract_small, concrete_unique, abstract_unique
bs=$4

if [[ $dataset == 'icse' ]]; then
    arr=(${datatype//_/ })
    type=${arr[0]}
    size=${arr[1]}
    input_base_path=/home/saikatc/Research/OpenNMT-py/rule_based_data/raw/$datasize/$datatype/test_new/
    model_path=/home/saikatc/Research/OpenNMT-py/rule_based_models/$datasize.$type.$size.token-best-acc.pt
else
    input_base_path=/home/saikatc/Research/OpenNMT-py/c_data/raw/$datasize/$datatype/test_new/
    model_path=/home/saikatc/Research/OpenNMT-py/c_models/$datasize.$datatype.token-best-acc.pt
fi
echo $input_base_path
echo $model_path
command='python translate_token_only.py -model '$model_path'
        -src '$input_base_path'prev.token -tgt '$input_base_path'next.token
        --name '$dataset'.'$datasize'.'$datatype' -gpu 0 -beam_size '$bs' -n_best '$bs' -verbose -replace_unk'
#        -verbose
echo $command