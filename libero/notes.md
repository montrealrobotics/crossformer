```bash
python libero_eval.py\
  --execute_all_actions\
  --log_file_path /network/scratch/o/ozgur.aslan/data/libero/finetuned/all_actions/50k/eval.log\
  --video-out-path /network/scratch/o/ozgur.aslan/data/libero/finetuned/all_actions/50k\
  --model_path /network/scratch/o/ozgur.aslan/cross_ft/crossformer_finetune/experiment_20250710_170012\
  --model_step 50000
```


```bash
python libero_eval.py\
  --ensemble\
  --log_file_path /network/scratch/o/ozgur.aslan/data/libero/finetuned/ensemble/30k/test.log\
  --video-out-path /network/scratch/o/ozgur.aslan/data/libero/finetuned/ensemble/30k\
  --model_path /network/scratch/o/ozgur.aslan/cross_ft/crossformer_finetune/experiment_20250710_170012\
  --model_step 30000
```