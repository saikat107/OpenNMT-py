#!/usr/bin/env bash
#python train.py -data rule_based_data/processed/all/original.small.token -save_model rule_based_model/all.original.small.token -valid_steps 500 -gpuid 0 --type token -report_every 500 -train_steps 20000

#python train.py -data rule_based_data/processed/all/original.small.augmented.token -save_model rule_based_model/all.original.small.augmented.token -valid_steps 500 -gpuid 0 --type token -report_every 500 -train_steps 20000

#python train.py -data rule_based_data/processed/filtered/original.small.token -save_model rule_based_model/filtered.original.small.token -valid_steps 500 -gpuid 0 --type token -report_every 500 -train_steps 20000

#python train.py -data rule_based_data/processed/filtered/original.small.rule -save_model rule_based_model/filtered.original.small.rule -valid_steps 500 -gpuid 0 --type tree -report_every 500 -train_steps 20000

#python train.py -data rule_based_data/processed/filtered/original.small.augmented.token -save_model rule_based_model/filtered.original.small.augmented.token -valid_steps 500 -gpuid 0 --type token -report_every 500 -train_steps 20000

for type in concrete original; do
	for tr in all filtered; do
		for size in small; do
			for kind in rule augmented.token; do
				echo $tr' '$type' '$size' '$kind
			    	python train.py -data rule_based_data/processed/$tr/$type.$size.$kind -save_model rule_based_models/$tr.$type.$size.$kind -gpuid 0 --type token
			done
		done
	done
done




