for type in concrete; do
	for size in small medium; do
		for kind in token rule augmented.token; do
			for tr in all; do
				echo $tr' '$type' '$size' '$kind
			        python train.py -data rule_based_data/processed/$tr/$type.$size.$kind \
			        -save_model rule_based_model/$tr.$type.$size.$kind \
			        -valid_steps 500 \
			        -gpuid 0 \
			        --type token \
			        -report_every 500 \
			        -train_steps 20000
			done
		done
	done
done





