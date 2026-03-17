from ruamel.yaml import YAML
from raisimGymTorch.env.bin import rsg_a1_task
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import io
import os
import math
import time
import numpy as np
import torch
import sys
np.set_printoptions(suppress=True, precision=3)

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."
base_dir = sys.argv[1]
runid = sys.argv[2]

# config
cfg = YAML().load(open(sys.argv[1] + "/cfg.yaml", 'r'))
cfg['environment']['num_envs'] = 1
cfg['environment']['num_threads'] = 1
cfg['environment']['render'] = True  # Enable visualization

cfg['environment']['test'] = True
# Uncomment this for more controlled tests
#cfg['environment']['randomize_friction'] = False
#cfg['environment']['randomize_mass'] = False
#cfg['environment']['randomize_motor_strength'] = False
#cfg['environment']['randomize_gains'] = False
#cfg['environment']['speedTest'] = False

# create environment from the configuration file
yaml = YAML()
cfg_stream = io.StringIO()
yaml.dump(cfg['environment'], cfg_stream)
env_cfg = cfg_stream.getvalue()
env = VecEnv(rsg_a1_task.RaisimGymEnv(home_path + "/rsc", env_cfg), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
policy_load_path = '/'.join([base_dir, 'policy_' + runid + '.pt'])
env.load_scaling(base_dir, int(runid))
loaded_graph = torch.jit.load(policy_load_path)


foot_contacts = []
eplen = 100

print("Visualizing and evaluating the current policy")
print("Make sure RaisimUnity OpenGL is running first!")

# Start video recording (optional - uncomment to record)
env.start_video_recording("policy_" + runid + ".mp4")

try:
    for update in range(1):
        env.reset()
        env.turn_on_visualization()

        eplen = 0
        # An high number, assumes curriculum is finished
        env.set_itr_number(30000) #int(runid))
        for step in range(5000):  # Reduced from 50000 for faster testing
            time.sleep(0.01)
            obs = env.observe(False)
            with torch.no_grad():
                action_ll = loaded_graph(torch.from_numpy(obs).cpu())
            action = action_ll.cpu().detach().numpy()

            reward_ll, dones = env.step(action)
            reward_info = env.get_reward_info()[0]
            eplen+=1
            if dones[0]:
                print(f"Episode completed with {eplen} steps")
                eplen = 0
                # You can put plotting stuff here!

        env.turn_off_visualization()
finally:
    # Always save video even if interrupted
    print("Saving video...")
    env.stop_video_recording()
    print(f"Video saved as policy_{runid}.mp4")

