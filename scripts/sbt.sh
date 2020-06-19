#!/usr/bin/env bash
if [[ $# != 4 ]]; then
  echo "Must provide 2 arguments "
  echo "dataset<code_change_data, pull_request_data>, <beam_size>, <input token/sbt> <output token/sbt>"
  exit
fi

dataset=$1 # icse, codit
bs=$2
input=$3
output=$4
input_base_path="/home/saikatc/Research/OpenNMT-py/data/raw/${dataset}/test/"
model_path="/home/saikatc/Research/OpenNMT-py/models/${dataset}/${input}2${output}-best-acc.pt"
echo $input_base_path
echo $model_path
python translate_token_only.py \
        -model ${model_path} \
        -src ${input_base_path}prev.${input} \
        -tgt ${input_base_path}next.${output} \
        --name ${dataset}/${input}2${output} \
        -beam_size ${bs} \
        -n_best ${bs} \
        -replace_unk \
        -verbose \
        -batch_size 64;
