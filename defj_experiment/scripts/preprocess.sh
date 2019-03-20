#!/usr/bin/env bash
#!/usr/bin/env bash
if [[ ! -d defj_experiment/data/processed ]]; then
    mkdir defj_experiment/data/processed
fi

if [[ ! -d defj_experiment/data/processed ]]; then
    mkdir defj_experiment/data/processed
fi


for fil in br original; do
        for kind in rule token augmented.token;do
            echo $fil' '$kind
            python preprocess.py \
            -train_src defj_experiment/data/raw/$fil/train/prev.$kind \
            -train_tgt defj_experiment/data/raw/$fil/train/next.$kind \
            -valid_src defj_experiment/data/raw/$fil/valid/prev.$kind \
            -valid_tgt defj_experiment/data/raw/$fil/valid/next.$kind \
            -save_data defj_experiment/data/processed/$fil.$kind \
            -share_vocab
        done
done
