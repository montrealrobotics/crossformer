## Thanks GEMINI 


import subprocess
import time
import sys
import os 

# Define the server and client script names
## funny enough we can also add argparser or dataclass to pass these arguments 
current_dir = os.path.dirname(__file__)
SERVER_SCRIPT = os.path.join(current_dir, "server.py")
CLIENT_SCRIPT = os.path.join(current_dir, "..", "libero", "libero_eval.py") 

out_path = "/network/scratch/o/ozgur.aslan/data/libero/client_server2"
model_path = "/network/scratch/o/ozgur.aslan/cross_ft/crossformer_finetune/experiment_20250710_170012"
model_step = "30000"

# 1. Start the server process using Popen (non-blocking)
#    - We redirect stdout and stderr to files for logging.
print(f"Starting server: {SERVER_SCRIPT}...")
with open('server.log', 'wb') as logfile:
    server_process = subprocess.Popen(
        [
         sys.executable, 
         SERVER_SCRIPT,
         "--model_path", model_path,
         "--model_step", model_step
        ], # Use sys.executable to ensure same Python interpreter
        stdout=logfile,
        stderr=logfile
    )

print(f"Server started with PID: {server_process.pid}")

try:
    # 2. Give the server a minute to start up and load crossformer model
    print("Waiting for 60 seconds...")
    time.sleep(60)

    # 3. Run the client process and wait for it to complete (blocking)
    #    - subprocess.run() is a blocking call by default.
    print(f"Starting client: {CLIENT_SCRIPT}...")
    client_result = subprocess.run([sys.executable, 
                                    CLIENT_SCRIPT, 
                                    "--out_path",
                                    out_path,
                                    "--execute_all_actions"])
    
    # Optional: Check if the client ran successfully
    if client_result.returncode == 0:
        print("Client finished successfully.")
    else:
        print(f"Client failed with return code: {client_result.returncode}")

finally:
    # 4. Terminate the server process
    #    - This block runs whether the client succeeded or failed.
    print(f"Terminating server with PID: {server_process.pid}...")
    server_process.terminate()  # Sends a graceful shutdown signal (SIGTERM)
    server_process.wait()       # Waits for the process to actually terminate
    print("Server terminated.")

print("Orchestration complete.")