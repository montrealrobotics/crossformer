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

If you get ```KeyError: 'getgrgid(): gid not found:```  error:  
Changed the line:
```python 
group = grp.getgrgid(st.st_gid).gr_name
```
to
```python
try:
    group = grp.getgrgid(st.st_gid).gr_name
except KeyError:
    group = str(st.st_gid)  # fallback to just the GID string
```
in ```.conda/envs/<<env_name>>/lib/python3.10/site-packages/etils/epath/backend.py```


## Install
- To install follow crossformer installing instructions.
- Installing server-client dependencies: ```pip install -r server_client_req.txt ```
- Checking if jax can see the gpu(s): ```import jax; jax.devices() ```
- Checking if tensorflow can see the gpu(s): ```import tensorflow as tf; tf.config.list_physical_devices('GPU') ```
- If ```tf.config.list_physical_devices('GPU')``` returns an empty list, try: ```pip install tensorflow[and-cuda]==2.15.1 ```
- To use libero:
  - Installing libero dependencies: ```pip install -r libero/libero_requirements.txt ```
  - Also need to install torch: ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ```