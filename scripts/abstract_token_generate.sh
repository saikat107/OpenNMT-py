#!/usr/bin/env bash
if [[ $# != 2 ]]; then
    echo "Must provide 2 arguments "
    echo "dataset<code_change_data, pull_request_data>, <beam_size>"
    exit
fi

dataset=$1 # icse, codit
#datasize=$2 # all, filtered, original
#datatype=$3 # concrete, abstract, concrete_small, abstract_small, concrete_unique, abstract_unique
bs=$2
input_base_path="/home/saikatc/Research/OpenNMT-py/data/raw/${dataset}/test/";
model_path="/home/saikatc/Research/OpenNMT-py/models/${dataset}/abstract.code-best-acc.pt"
#if [[ $dataset == 'icse' ]]; then
##    arr=(${datatype//_/ })
##    type=${arr[0]}
##    size=${arr[1]}
##    input_base_path=/home/saikatc/Research/OpenNMT-py/rule_based_data/raw/$datasize/$datatype/test/
##    model_path=/home/saikatc/Research/OpenNMT-py/rule_based_model/$datasize.$type.$size.token-best-acc.pt
#
#else
#    input_base_path=/home/saikatc/Research/OpenNMT-py/c_data/raw/$datasize/$datatype/test/
#    model_path=/home/saikatc/Research/OpenNMT-py/c_models/$datasize.$datatype.token-best-acc.pt
#fi
echo $input_base_path
echo $model_path
command='python translate_token_only.py -model '$model_path'
        -src '$input_base_path'prev.abstract.code -tgt '$input_base_path'next.abstract.code
        --name '$dataset'/abstract.code -gpu 0 -beam_size '$bs' -n_best '$bs' -replace_unk -verbose -batch_size 64'
#        -verbose
echo $command
$command