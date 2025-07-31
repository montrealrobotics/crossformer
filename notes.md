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
- To use widowx robot:
  - Crossformer should be installed in a server pc (green)
  - Run crossformer server ```pyhon scripts/server.py --env_config widowx```   
This will start the server and it will query the policy three times to compile the model. (This takes a bit of time so wait until you see the last passed time print)
  - Don't forget to get the server's ip (will be used in client)
  - On the robot's control pc (probably a nuc):
    - Install (if it is not already) bridge_data_robot ```git clone https://github.com/montrealrobotics/bridge_data_robot.git --single-branch -b nuc_logicam```
      - Following the instructions in the readme under Setup to build the docker image for robot control server. 
      - You do not need to install ros or nvidia-docker but need to install docker, docker-compose and widowx drivers (it will only install udev/rules)
    - Create a conda env: ```conda create --name cf python=3.10```
    - Clone crossformer repo: ```git clone https://github.com/montrealrobotics/crossformer.git```
    - Go to crossformer directory and install widowx dependencies: ```pip install -r widowx/widowx_requirements.txt``` and install crossformer and cf_scripts: ```pip install -e .```
    - Go to bridge_data_robot/widowx_envs directory and install it: ```pip install -e .```
    - Open three terminals:
      - In the first run (if you are not already): ```USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up robonet```
      - In the second run (if you are not already): ```docker compose exec robonet bash -lic "widowx_env_service --server"```
      - In the third run: ```python widowx/widowx_eval.py --policy_ip <ip of server that runs crossformer> --num_timesteps 100 --video_save_path widowx/video``` (video_save_path argument is optinal to save the video of the policy execution)
    - Always run ```docker compose exec robonet bash -lic "go_sleep"``` before stopping the robonet_widowx running in the first terminal (Otherwise robot falls abruptly) 
