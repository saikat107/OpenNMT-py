for data_name in pull_request_data code_change_data; do
  echo ${data_name}
  python preprocess.py -train_src data/raw/${data_name}/train/prev.sbt \
                  -train_tgt data/raw/${data_name}/train/next.token \
                  -valid_src data/raw/${data_name}/train/prev.sbt \
                  -valid_tgt data/raw/${data_name}/train/next.token \
                  -save_data data/processed/${data_name}/sbt2tok;

  python preprocess.py -train_src data/raw/${data_name}/train/prev.token \
                  -train_tgt data/raw/${data_name}/train/next.sbt \
                  -valid_src data/raw/${data_name}/train/prev.token \
                  -valid_tgt data/raw/${data_name}/train/next.sbt \
                  -save_data data/processed/${data_name}/tok2sbt;

  python preprocess.py -train_src data/raw/${data_name}/train/prev.sbt \
                  -train_tgt data/raw/${data_name}/train/next.sbt \
                  -valid_src data/raw/${data_name}/train/prev.sbt \
                  -valid_tgt data/raw/${data_name}/train/next.sbt \
                  -save_data data/processed/${data_name}/sbt2sbt;
done