bs=$1
python translate_token_only.py -model c_models/all.concrete.token-best-acc.pt -src c_data/raw/all/concrete/test_new/prev.token -tgt c_data/raw/all/concrete/test_new/next.token -gpu 0 -beam_size $bs -n_best $bs --name codit.all.token.top.$bs -verbose -replace_unk -batch_size 50
