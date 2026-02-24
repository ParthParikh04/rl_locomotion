"""
Example training script using Gymnasium environment with PPO algorithm.
This demonstrates how to replace the raisim environment with any gymnasium environment
while keeping the same PPO algorithm intact.
"""

from raisimGymTorch.env.GymnasiumVecEnv import GymnasiumVecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import time
import argparse
import os
import math

try:
    import wandb
except ImportError:
    wandb = None

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="Humanoid-v4", 
                    help="Gymnasium environment ID (e.g., 'Humanoid-v4', 'Walker2d-v4')")
parser.add_argument("--num-envs", type=int, default=4, 
                    help="Number of parallel environments")
parser.add_argument("--num-steps", type=int, default=2000, 
                    help="Number of steps per update")
parser.add_argument("--num-updates", type=int, default=10000, 
                    help="Number of training updates")
parser.add_argument("--gpu", type=int, default=0, 
                    help="GPU ID to use")
parser.add_argument("--debug", action="store_true", 
                    help="Run in debug mode (single env, CPU)")
parser.add_argument("--log-dir", type=str, default="./logs", 
                    help="Directory for logs and checkpoints")
parser.add_argument("--seed", type=int, default=42, 
                    help="Random seed")
parser.add_argument("--name", type=str, default=None,
                    help="Name for the run")

args = parser.parse_args()

# Set random seeds
rng_seed = args.seed
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

# Device configuration
if args.debug:
    num_envs = 1
    device_type = 'cpu'
else:
    num_envs = args.num_envs
    device_type = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device_type}")
print(f"Environment: {args.env_id}")
print(f"Number of parallel environments: {num_envs}")

# Create output directory
os.makedirs(args.log_dir, exist_ok=True)

# Create gymnasium environment wrapper
try:
    env = GymnasiumVecEnv(
        env_id=args.env_id,
        num_envs=num_envs,
        normalize_ob=True,
        seed=rng_seed,
        normalize_rew=True,
        clip_obs=10.0
    )
except Exception as e:
    print(f"\nError creating environment: {e}")
    print(f"\nNote: Make sure the environment is available. Install dependencies if needed:")
    print("  - For MuJoCo environments: pip install gymnasium[mujoco]")
    print("  - For Box2D environments: pip install swig gymnasium[box2d]")
    exit(1)

# Get environment dimensions
ob_dim = env.num_obs
act_dim = env.num_acts

# For small observation spaces, PPO will double storage dimensions
# So we need to duplicate observations accordingly 
# PPO will see network obs_shape and allocate_storage as 2x if < 200
storage_ob_dim = ob_dim * 2 if ob_dim < 200 else ob_dim

print(f"Observation dimension: {ob_dim}")
print(f"Action dimension: {act_dim}")
print(f"Storage observation dimension: {storage_ob_dim}")

# Create actor and critic networks
init_var = 0.3

# Create networks with raw observation dimension
# PPO will handle storage doubling internally
actor = ppo_module.Actor(
    ppo_module.MLP(
        [256, 256],
        nn.LeakyReLU,
        ob_dim,
        act_dim
    ),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, init_var),
    device_type
)

critic = ppo_module.Critic(
    ppo_module.MLP(
        [256, 256],
        nn.LeakyReLU,
        ob_dim,
        1
    ),
    device_type
)

# Training parameters
n_steps = args.num_steps
num_updates = args.num_updates
num_learning_epochs = 4
num_mini_batches = 4
learning_rate = 5e-4
gamma = 0.999
lam = 0.95

# Create PPO trainer
ppo = PPO.PPO(
    actor=actor,
    critic=critic,
    num_envs=num_envs,
    num_transitions_per_env=n_steps,
    num_learning_epochs=num_learning_epochs,
    num_mini_batches=num_mini_batches,
    device=device_type,
    log_dir=args.log_dir,
    mini_batch_sampling='in_order',
    learning_rate=learning_rate,
    gamma=gamma,
    lam=lam,
    clip_param=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.0,
    max_grad_norm=0.5
)

if wandb and not args.debug:
    wandb.init(project='rl-gymnasium', config=args.__dict__, name=args.name)
    wandb.watch(actor.architecture, log_freq=100)
    wandb.watch(critic.architecture, log_freq=100)

print("\n" + "="*80)
print("Starting training...")
print("="*80 + "\n")

# Training loop
total_steps = 0
episode_rewards = []

# Calculate duplication needed to match storage dimension expectations
duplication_factor = storage_ob_dim // ob_dim

def duplicate_observations(obs):
    """Duplicate observations to match PPO storage expectations."""
    if duplication_factor == 1:
        return obs
    result = obs
    for _ in range(duplication_factor - 1):
        result = np.concatenate([result, obs], axis=1)
    return result

try:
    for update in range(num_updates):
        start_time = time.time()
        
        # Reset environment
        env.reset()
        update_reward_sum = 0.0
        update_done_count = 0
        
        # Collect trajectories
        for step in range(n_steps):
            # Get observations and duplicate to match network dimension
            obs_raw = env.observe(update_mean=True)
            obs = duplicate_observations(obs_raw)
            
            # Get action and log probability from policy
            action = ppo.observe(obs)
            
            # Step environment
            reward, dones = env.step(action)
            
            # Add transitions to storage
            ppo.step(value_obs=obs, rews=reward, dones=dones, infos=[])
            
            update_reward_sum += np.sum(reward)
            update_done_count += np.sum(dones)
        
        # Compute value for next state and update policy
        obs_raw = env.observe(update_mean=True)
        obs = duplicate_observations(obs_raw)
        ppo.update(
            actor_obs=obs,
            value_obs=obs,
            log_this_iteration=update % 10 == 0,
            update=update
        )
        
        elapsed_time = time.time() - start_time
        total_steps += n_steps * num_envs
        avg_reward = update_reward_sum / (update_done_count + 1e-6)
        
        if update % 10 == 0:
            print(f"Update {update:5d} | Total Steps: {total_steps:8d} | "
                  f"Avg Reward: {avg_reward:8.2f} | Time: {elapsed_time:6.2f}s")
            
            if wandb:
                wandb.log({
                    "update": update,
                    "avg_reward": avg_reward,
                    "total_steps": total_steps
                })
        
        # Save checkpoint
        if (update + 1) % 100 == 0:
            checkpoint_path = os.path.join(args.log_dir, f"checkpoint_update_{update}.pt")
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'update': update,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user")

# Save final model
final_checkpoint = os.path.join(args.log_dir, "final_checkpoint.pt")
torch.save({
    'actor_architecture_state_dict': actor.architecture.state_dict(),
    'actor_distribution_state_dict': actor.distribution.state_dict(),
    'critic_architecture_state_dict': critic.architecture.state_dict(),
    'optimizer_state_dict': ppo.optimizer.state_dict(),
}, final_checkpoint)
print(f"\nSaved final checkpoint to {final_checkpoint}")

# Clean up
env.close()
if wandb:
    wandb.finish()

print("Training completed!")

