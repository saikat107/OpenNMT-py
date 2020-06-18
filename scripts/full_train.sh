for data_name in pull_request_data code_change_data; do
  for kind in token rule augmented.token abstract.code sbt2tok tok2sbt sbt2sbt; do
    echo "Training on ${data_name}-${kind}"
    python train.py \
          -data data/processed/${data_name}/$kind \
          -save_model models/${data_name}/$kind \
          --type token \
          -valid_steps 500 \
          -gpuid 0 \
          --type token \
          -report_every 500 \
          -train_steps 100000;
  done
done