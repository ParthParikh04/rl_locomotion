#!/usr/bin/env python3
"""
Workaround for broken native video recording. 
Captures screenshots during simulation and assembles them into a video using ffmpeg.
"""
import os
import sys
import subprocess
import glob
import shutil
import time

from ruamel.yaml import YAML
from raisimGymTorch.env.bin import rsg_a1_task
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import io
import numpy as np
import torch

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."
base_dir = sys.argv[1]
runid = sys.argv[2]

# config
cfg = YAML().load(open(sys.argv[1] + "/cfg.yaml", 'r'))
cfg['environment']['num_envs'] = 1
cfg['environment']['num_threads'] = 1
cfg['environment']['render'] = True
cfg['environment']['test'] = True

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
n_steps = int(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
policy_load_path = '/'.join([base_dir, 'policy_' + runid + '.pt'])
env.load_scaling(base_dir, int(runid))
loaded_graph = torch.jit.load(policy_load_path)

foot_contacts = []
eplen = 100

print("Visualizing and evaluating the current policy")
print("Make sure RaisimUnity OpenGL is running first!")

# Create temporary screenshot directory
screenshot_dir = '/tmp/raisim_screenshots'
if os.path.exists(screenshot_dir):
    shutil.rmtree(screenshot_dir)
os.makedirs(screenshot_dir)

output_video = f"policy_{runid}.mp4"

try:
    for update in range(1):
        env.reset()
        env.turn_on_visualization()

        eplen = 0
        frame_count = 0
        capture_interval = 5  # Capture every 5 steps (for 200 FPS sim -> ~40 FPS video)
        
        # An high number, assumes curriculum is finished
        env.set_itr_number(30000)
        
        for step in range(5000):
            obs = env.observe(False)
            with torch.no_grad():
                action_ll = loaded_graph(torch.from_numpy(obs).cpu())
            action = action_ll.cpu().detach().numpy()

            reward_ll, dones = env.step(action)
            reward_info = env.get_reward_info()[0]
            eplen += 1
            
            # Capture screenshot periodically
            if step % capture_interval == 0:
                env.env.wrapper.server.requestSaveScreenshot()  # Request screenshot
                frame_count += 1
                time.sleep(0.001)  # Small delay for screenshot to be saved

            if dones[0]:
                print(f"Episode completed with {eplen} steps")
                eplen = 0

        env.turn_off_visualization()
        
        # Wait a moment for final screenshots to be saved
        time.sleep(1)
        
        # Use ffmpeg to assemble screenshots into video
        print(f"\nAssembling {frame_count} screenshots into video...")
        
        # Find all screenshots
        screenshots = sorted(glob.glob('/home/pparikh47/projects/cs7641-project/raisimLib/raisimUnityOpengl/linux/Screenshot/*.png'))
        
        if len(screenshots) > 0:
            # Use latest screenshots (they should be from our recording session)
            latest_screenshots = screenshots[-frame_count:] if len(screenshots) >= frame_count else screenshots
            
            # Create ffmpeg command
            # Use the first screenshot to determine resolution
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-framerate', '40',  # 40 FPS video
                '-pattern_type', 'glob',
                '-i', '/home/pparikh47/projects/cs7641-project/raisimLib/raisimUnityOpengl/linux/Screenshot/Screenshot-*.png',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-start_number', '0',
                output_video
            ]
            
            print(f"Running: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, cwd=screenshot_dir, capture_output=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully created {output_video}")
                print(f"  Location: {os.path.join(screenshot_dir, output_video)}")
                # Copy to original location
                shutil.copy(os.path.join(screenshot_dir, output_video), 
                           '/home/pparikh47/projects/cs7641-project/raisimLib/raisimUnityOpengl/linux/Screenshot/')
            else:
                print(f"✗ ffmpeg failed: {result.stderr.decode()}")
        else:
            print("✗ No screenshots found!")

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    print("Recording session ended.")
