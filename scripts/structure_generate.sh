#!/usr/bin/env bash
#if [[ $# != 3 ]]; then
#    echo "Must provide 3 arguments "
#    echo "dataset<icse, codit>, ditasize<all, filtered>, datatype<concrete, abstract, concrete_small, abstract_small, concrete_unique, abstract_unique>"
#    exit
#fi
#
#dataset=$1 # icse, codit
#datasize=$2 # all, filtered, original
#datatype=$3 # concrete, abstract, concrete_small, abstract_small, concrete_unique, abstract_unique
#
#if [[ $dataset == 'icse' ]]; then
#    arr=(${datatype//_/ })
#    type=${arr[0]}
#    size=${arr[1]}
#    input_base_path=/home/saikatc/Research/OpenNMT-py/rule_based_data/raw/$datasize/$datatype/test/
#    model_path=/home/saikatc/Research/OpenNMT-py/rule_based_model/$datasize.$type.$size.rule-best-acc.pt
#    grammar_path=/home/saikatc/Research/OpenNMT-py/rule_based_data/raw/$datasize/$datatype/grammar.bin
#else
#    input_base_path=/home/saikatc/Research/OpenNMT-py/c_data/raw/$datasize/$datatype/test/
#    model_path=/home/saikatc/Research/OpenNMT-py/c_models/$datasize.$datatype.rule-best-acc.pt
#    grammar_path=/home/saikatc/Research/OpenNMT-py/c_data/raw/$datasize/$datatype/grammar.bin
#fi

if [[ $# -le 1 ]]; then
    echo "Must provide 2 arguments "
    echo "dataset<code_change_data, pull_request_data>, <beam_size>"
    exit
fi

dataset=$1 # icse, codit
bs=$2
nbest=$bs
if [[ $# == 3 ]]; then
  nbest=$3
fi
#datasize=$2 # all, filtered, original
#datatype=$3 # concrete, abstract, concrete_small, abstract_small, concrete_unique, abstract_unique
input_base_path="/home/saikatc/Research/OpenNMT-py/data/raw/${dataset}/test/";
model_path="/home/saikatc/Research/OpenNMT-py/models/${dataset}/rule-best-acc.pt"
grammar_path="/home/saikatc/Research/OpenNMT-py/data/raw/${dataset}/grammar.bin"


#echo $input_base_path
echo $model_path ' '$bs
python translate_structure.py -model $model_path \
        -src $input_base_path/prev.rule -tgt $input_base_path/next.rule\
        --name ${dataset}/structure-only \
        --grammar $grammar_path -beam_size $bs -n_best $nbest -gpu 0 \
#        -verbose