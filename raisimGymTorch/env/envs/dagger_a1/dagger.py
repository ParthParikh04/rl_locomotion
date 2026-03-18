from ruamel.yaml import YAML
from raisimGymTorch.env.bin import dagger_a1
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
import os
import io
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
from raisimGymTorch.algo.ppo.dagger import DaggerExpert, DaggerAgent, DaggerTrainer
import torch.nn as nn
import numpy as np
import torch
import argparse
try:
    import wandb
except:
    wandb = None

parser = argparse.ArgumentParser()
parser.add_argument("--exptid", type = int, help='experiment id to prepend to the run')
parser.add_argument("--overwrite", action = 'store_true')
parser.add_argument("--override", action = 'store_true')
parser.add_argument("--debug", action = 'store_true')
parser.add_argument("--loadpth", type = str, default = None)
parser.add_argument("--loadid", type = str, default = None)
parser.add_argument("--gpu", type = int, default = 0)
parser.add_argument("--name", type = str)
parser.add_argument("--ext_act", type = str, default='leakyRelu')
parser.add_argument("--method", type = str, default='supervised', choices=['supervised', 'unsupervised'])
parser.add_argument("--z_method", type = str, default='predictive',
                    choices=['predictive', 'pca', 'kmeans', 'gmm', 'dbscan'])
parser.add_argument("--use_priv_decoder", action='store_true')
parser.add_argument("--priv_decoder_dim", type=int, default=23)
parser.add_argument("--priv_decoder_weight", type=float, default=1.0)
parser.add_argument("--policy_loss_weight", type=float, default=0.0)
parser.add_argument("--allow_policy_grad_to_z", action='store_true')
parser.add_argument("--train_student_mlp", action='store_true')
args = parser.parse_args()


# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."


# config
cfg = YAML().load(open(args.loadpth + "/cfg.yaml", 'r'))
with open("cfg.yaml", 'r') as f:
    dagger_cfg = YAML().load(f)
# set seed
rng_seed = cfg['seed']
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

t_steps = dagger_cfg['environment']['history_len']
base_dims = cfg['environment']['baseDim']
n_futures = int(dagger_cfg['environment']['n_futures'])
cfg['environment']['n_futures'] = n_futures
prop_latent_dim = 8
geom_latent_dim = 1

activation_fn_map = {'none': None, 'tanh': nn.Tanh, 'leakyRelu': nn.LeakyReLU}
output_activation_fn = activation_fn_map[cfg['architecture']['activation']]
small_init_flag = cfg['architecture']['small_init']
ext_activation_map = activation_fn_map[args.ext_act]

if args.debug:
    cfg['environment']['num_envs'] = 1
    cfg['environment']['num_threads'] = 1
    device_type = 'cpu'
else:
    device_type = 'cuda:{}'.format(args.gpu)

cfg['environment']['num_threads'] = dagger_cfg['environment']['num_threads']

cfg['environment']['history_len'] = t_steps
cfg['environment']['test'] = False
cfg['environment']['eval'] = False

# create environment from the configuration file
yaml = YAML()
yaml_stream = io.StringIO()
yaml.dump(cfg['environment'], yaml_stream)
env = VecEnv(dagger_a1.RaisimGymEnv(home_path + "/rsc", yaml_stream.getvalue()), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts


priv_dim = ob_dim - base_dims * (t_steps + 1)

# save a few logs about the run
cfg['environment']['loadpth'] = args.loadpth
cfg['environment']['loadid'] = args.loadid


# save the configuration and other files
saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/dagger_ckpt/" + '{:04d}'.format(args.exptid),
                           save_items=[task_path + "/Environment.hpp", os.path.join(args.loadpth, f"policy_{args.loadid}.pt")],
                                       config = cfg, overwrite = (args.overwrite or args.override))

if wandb:
    wandb.init(project='command_loco', config=dict(cfg), name="dagger/" + args.name)
    wandb.save(home_path + '/raisimGymTorch/env/envs/rsg_a1_task/Environment.hpp')

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

expert_policy = DaggerExpert(args.loadpth, args.loadid, ob_dim, t_steps,
                             base_dims, env.obs_rms.mean.shape[0],
                             geomDim=int(cfg['environment']['geomDim']),
                             n_futures=n_futures)
student_mlp = ppo_module.MLP(cfg['architecture']['policy_net'],
                                        ext_activation_map,
                                        base_dims + prop_latent_dim + (n_futures+1)*geom_latent_dim,
                                        act_dim,
                                        output_activation_fn, 
                                        small_init_flag)
prop_latent_encoder = ppo_module.StateHistoryEncoder(ext_activation_map, base_dims, t_steps,
                                                     prop_latent_dim + (n_futures+1)*geom_latent_dim,
                                                     predict_next_state=(args.method == 'unsupervised'),
                                                     next_state_dim=base_dims)

actor = DaggerAgent(expert_policy,
                    prop_latent_encoder,
                    student_mlp, t_steps, base_dims, device_type,
                    n_futures=n_futures,
                    train_student_mlp=args.train_student_mlp)

dagger = DaggerTrainer(
              actor=actor,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              obs_shape = ob_dim,
              latent_shape = prop_latent_dim + geom_latent_dim*(n_futures+1),
              base_obs_size=base_dims,
              history_len=t_steps,
              num_learning_epochs=1,
              num_mini_batches=4,
              device=device_type,
              learning_rate=5e-3,
              method=args.method,
              z_method=args.z_method,
              use_priv_decoder=args.use_priv_decoder,
              priv_decoder_dim=args.priv_decoder_dim,
              priv_decoder_weight=args.priv_decoder_weight,
              policy_loss_weight=args.policy_loss_weight,
              detach_z_for_policy=(not args.allow_policy_grad_to_z),
              random_state=rng_seed,
              )

env.obs_rms.mean = actor.mean
env.obs_rms.var = actor.var

penalty_scale = np.array([cfg['environment']['lateralVelRewardCoeff'], cfg['environment']['angularVelRewardCoeff'], cfg['environment']['deltaTorqueRewardCoeff'], cfg['environment']['actionRewardCoeff'], cfg['environment']['sidewaysRewardCoeff'], cfg['environment']['jointSpeedRewardCoeff'], cfg['environment']['deltaContactRewardCoeff'], cfg['environment']['deltaReleaseRewardCoeff'], cfg['environment']['footSlipRewardCoeff'], cfg['environment']['upwardRewardCoeff'], cfg['environment']['workRewardCoeff'], cfg['environment']['yAccRewardCoeff'], 1., 1., 1.])


env.set_itr_number(int(args.loadid))

for update in range(1201):
    start = time.time()
    env.reset()
    dagger.reset_rollout()
    reward_ll_sum = 0
    forwardX_sum = 0
    penalty_sum = 0
    done_sum = 0
    average_dones = 0.

    if update %  dagger_cfg['environment']['eval_every_n'] == 0:
        print("Saving the current policy")
        actor.save_deterministic_graph(saver.data_dir+"/prop_encoder_"+str(update)+'.pt', 
                                       #saver.data_dir+"/geom_encoder_"+str(update)+'.pt', 
                                       saver.data_dir+"/mlp_"+str(update)+'.pt', 
                                       torch.rand(1, ob_dim - priv_dim + prop_latent_dim + (n_futures+1)*geom_latent_dim).cpu())

        env.save_scaling(saver.data_dir, str(update))

    # actual training
    for step in range(n_steps):
        obs = env.observe(update_mean=False)

        action = dagger.observe(obs)
        reward, dones = env.step(action)
        dagger.step(obs)
        unscaled_reward_info = env.get_reward_info()
        forwardX = unscaled_reward_info[:, 0]
        penalty = unscaled_reward_info[:, 1:]
        done_sum = done_sum + sum(dones)
        reward_ll_sum = reward_ll_sum + sum(reward)
        forwardX_sum += np.sum(forwardX)
        penalty_sum += np.sum(penalty, axis=0)

    env.curriculum_callback()

    # backward step
    loss_metrics = dagger.update()
    prop_mse_loss = loss_metrics['prop_mse']
    geom_mse_loss = loss_metrics['geom_mse']
    unsup_mse_loss = loss_metrics['unsup_mse']
    decoder_mse_loss = loss_metrics['decoder_mse']
    policy_mse_loss = loss_metrics['policy_mse']

    end = time.time()
    forwardX = forwardX_sum / total_steps
    forwardXReward = forwardX_sum * cfg['environment']['forwardVelRewardCoeff'] / total_steps

    forwardY, forwardZ, deltaTorque, action, sideways, jointSpeed, deltaContact, deltaRelease, footSlip, upward, work, yAcc, torqueSquare, stepHeight, walkedDist = penalty_sum / total_steps
    forwardYReward, forwardZReward, deltaTorqueReward, actionReward, sidewaysReward, jointSpeedReward, deltaContactReward, deltaReleaseReward, footSlipReward, upwardReward, workReward, yAccReward, torq, stepHeight, walkedDist = scaled_penalty = penalty_sum * penalty_scale / total_steps

    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)


    if wandb:
        wandb.log({'forwardX': forwardX, 
        'forwardX_reward': forwardXReward, 
        'forwardY': forwardY, 
        'forwardY_reward': forwardYReward, 
        'forwardZ': forwardZ, 
        'forwardZ_reward': forwardZReward, 
        'deltaTorque': deltaTorque, 
        'deltaTorque_reward': deltaTorqueReward, 
        'action': action, 
        'stepHeight': stepHeight,
        'action_reward': actionReward, 
        'sideways': sideways, 
        'sideways_reward': sidewaysReward, 
        'jointSpeed': jointSpeed, 
        'jointSpeed_reward': jointSpeedReward, 
        'deltaContact': deltaContact, 
        'deltaContact_reward': deltaContactReward, 
        'deltaRelease': deltaRelease, 
        'deltaRelease_reward': deltaReleaseReward, 
        'footSlip': footSlip, 
        'footSlip_reward': footSlipReward, 
        'upward': upward, 
        'upward_reward': upwardReward, 
        'work': work, 
        'work_reward': workReward, 
        'yAcc': yAcc, 
        'yAcc_reward': yAccReward,
        'torqueSquare': torqueSquare,
        'dones': average_dones,
        'walkedDist': walkedDist,
        'Method': 0 if args.method == 'supervised' else 1,
        'Latent Prop MSE': prop_mse_loss,
        'Latent Geom MSE': geom_mse_loss,
        'Unsupervised Next Obs MSE': unsup_mse_loss,
        'Priv Decoder MSE': decoder_mse_loss,
        'Policy MSE': policy_mse_loss})

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("method: ", args.method))
    print('{:<40} {:>6}'.format("z_method: ", args.z_method))
    print('{:<40} {:>6}'.format("use_priv_decoder: ", str(args.use_priv_decoder)))
    print('{:<40} {:>6}'.format("detach z for policy: ", str(not args.allow_policy_grad_to_z)))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("prop mse loss: ", '{:0.10f}'.format(prop_mse_loss)))
    print('{:<40} {:>6}'.format("geom mse loss: ", '{:0.10f}'.format(geom_mse_loss)))
    print('{:<40} {:>6}'.format("unsup next obs mse: ", '{:0.10f}'.format(unsup_mse_loss)))
    print('{:<40} {:>6}'.format("priv decoder mse: ", '{:0.10f}'.format(decoder_mse_loss)))
    print('{:<40} {:>6}'.format("policy mse: ", '{:0.10f}'.format(policy_mse_loss)))
    print('{:<40} {:>6}'.format("average forward reward: ", '{:0.10f}'.format(forwardXReward)))
    print('{:<40} {:>6}'.format("average penalty reward: ", ', '.join(['{:0.4f}'.format(r) for r in scaled_penalty])))
    print('{:<40} {:>6}'.format("average walked dist: ", '{:0.10f}'.format(scaled_penalty[-1])))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('----------------------------------------------------\n')
