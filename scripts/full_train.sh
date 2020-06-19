for data_name in pull_request_data; do
  for kind in token sbt2tok tok2sbt sbt2sbt; do
    echo "Training on ${data_name}-${kind}"
    python train.py \
          -data data/processed/${data_name}/$kind \
          -save_model models/${data_name}/$kind \
          --type token \
          -valid_steps 500 \
          -gpuid 0 \
          -batch_size 16 \
          --type token \
          -report_every 500 \
          -train_steps 50000;
  done
done