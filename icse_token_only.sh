bs=$1
python translate_token_only.py -model icse_models/all.concrete.small.token-best-acc.pt -src icse_data/raw/all/concrete_small/test_new/prev.token -tgt icse_data/raw/all/concrete_small/test_new/next.token -beam_size $bs -n_best $bs -gpu 0 --name icse.token.only.top.$bs
