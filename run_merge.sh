local_dir=$1

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir $local_dir \
  --target_dir $local_dir/huggingface  