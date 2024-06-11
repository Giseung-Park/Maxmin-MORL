import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl.environment.env import SumoEnvironment

from stable_baselines3.dqn.dqn import MaxminMFQ

import pdb
import random
import argparse
import numpy as np
import wandb

if __name__ == "__main__":
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Maxmin value-based MORL algorithm""")
    ### 1. Environment parameters

    prs.add_argument("-rd", dest="reward_dim", type=int, default=4, help="Reward dimension\n")
    prs.add_argument("-rdp", dest="r_dim_policy", type=int, default=1, help="equals 1\n")

    prs.add_argument("-dt", dest="delta_time", type=int, default=30, help="Action period\n")
    prs.add_argument("-yt", dest="yellow_time", type=int, default=4, help="Yellow light period\n")
    prs.add_argument("-ns", dest="num_seconds", type=int, default=9000, help="Total Seconds per Episode. "
                                                                               " We have total episodes of total_timesteps/(num_seconds/delta_time)\n")
    prs.add_argument("-tt", dest="total_timesteps", type=int, default=100000,
                     help="Total Timesteps. We have total episodes of total_timesteps/(num_seconds/delta_time).\n")

    prs.add_argument("-bf", dest="buffer_size", type=int, default=50000, help="Buffer size\n")
    prs.add_argument("-se", dest="seed", type=int, default=0, help="Random seed\n")

    ### 2. Algorithm parameters
    ## Main Q learning rate
    prs.add_argument("-mlr", dest="main_learning_rate", type=float, default=0.001,
                     help="Learning Rate of Main Q function\n")

    ## soft target update to incorporate updated w information
    prs.add_argument("-tgit", dest="target_update_interval", type=int, default=1, help="Target_update_interval\n")
    prs.add_argument("-tau", dest="tau", type=float, default=0.001, help="Soft Target update ratio\n")

    ## Exploration strategy for soft Q-update: alpha scheduling
    prs.add_argument("-alin", dest="ent_alpha_act_init", type=float, default=5.0,
                     help="Entropy coefficient for initial action selection. Less than ent_alpha.\n")
    prs.add_argument("-al", dest="ent_alpha", type=float, default=0.1, help="Entropy coefficient for training and final action selection\n")
    prs.add_argument("-alann", dest="annealing_step", type=int, default=10000, help="Length of Linear Entropy schedule."
                                                                                    "Less than total timesteps.\n")

    ## Perturbation parameters
    prs.add_argument("-init", dest="init_frac", type=float, default=0.0005,
                     help="Initialization ratio for Soft-Q Update\n")
    prs.add_argument("-pwlr", dest="perturb_w_learning_rate", type=float, default=0.01, help="Learning Rate of w\n")

    prs.add_argument("-perw", dest="period_cal_w_grad", type=int, default=1, help="Period of calculating w gradient\n")
    prs.add_argument("-pqlr", dest="perturb_q_learning_rate", type=float, default=0.001,
                     help="Learning Rate of each q_w\n")
    prs.add_argument("-pgst", dest="perturb_grad_step", type=int, default=1,
                     help="Number of gradient steps for perturbation\n")
    prs.add_argument("-pqnum", dest="perturb_q_copy_num", type=int, default=20,
                     help="Number of copied q networks for perturbation\n")
    prs.add_argument("-pstd", dest="perturb_std_dev", type=float, default=0.01,
                     help="Standard deviation for perturbed noise\n")

    ## Main Q grad step
    prs.add_argument("-mqst", dest="q_grad_st_after_init", type=int, default=3,
                     help="Number of gradient steps for main Q function after init state\n")

    def parse_input(arg):
        if ',' in arg:
            return [float(item) for item in arg.split(',')]
        elif isinstance(arg, str): # ['uniform', 'dirichlet']
            return arg
        else:
            raise NotImplementedError

    # weight initialize
    prs.add_argument("-winit", dest="weight_initialize", type=parse_input, nargs='?', default='uniform', help='Initialize Weight w')

    ## Option for w scheduling. For now, we set this as 'sqrt_inverse'
    prs.add_argument("-wsch", dest="w_schedule_option", type=str, choices=['sqrt_inverse', 'inverse', 'linear'],
                     default='sqrt_inverse', help="Option for w scheduling\n")

    ## Not used - epsilon greedy parameter
    prs.add_argument("-epinit", dest="exploration_initial_eps", type=float, default=0,
                     help="exploration_initial_eps\n")
    prs.add_argument("-epfin", dest="exploration_final_eps", type=float, default=0,
                     help="exploration_final_eps\n")
    prs.add_argument("-epfr", dest="exploration_fraction", type=float, default=0,
                     help="exploration_fraction\n")

    #Others: Weight decay in Adam. set as zero
    prs.add_argument("-wd", dest="weight_decay", type=float, default=0,
                     help="Weight for L2 regularization in Adam optimizer\n")

    prs.add_argument("-avwin", dest="stats_window_size", type=int, default=32,
                     help="The number of episodes to average\n")

    args = prs.parse_args()

    r_dim = args.reward_dim
    assert r_dim > 1

    ## wandb
    wandb.init(project="Maxmin_Traffic", job_type="train")
    wandb.run.name = "seed=" + str(args.seed)

    # random seed ## Already in set_random_seed in utils.py, but set randomness fixed in env
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = SumoEnvironment(
        net_file="nets/big-intersection/big-intersection.net.xml",
        route_file="nets/big-intersection/routes.rou_asym_10000_long.xml",
        single_agent=True,
        use_gui=False, #False
        delta_time=args.delta_time,
        yellow_time=args.yellow_time,
        num_seconds=args.num_seconds,
        sumo_seed=args.seed,
    )

    model = MaxminMFQ(
        env=env,
        policy="SQLPolicy",
        learning_rate=args.main_learning_rate,
        learning_starts=0,
        train_freq=1,
        target_update_interval=args.target_update_interval,
        tau=args.tau,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        exploration_fraction=args.exploration_fraction,
        verbose=1,
        seed=args.seed,
        r_dim=r_dim,
        r_dim_policy=1,
        buffer_size=args.buffer_size,
        ent_alpha=args.ent_alpha,
        weight_decay=args.weight_decay,
        ####### perturbation parameters
        soft_q_init_fraction=args.init_frac,
        perturb_w_learning_rate=args.perturb_w_learning_rate,
        period_cal_w_grad=args.period_cal_w_grad,
        perturb_q_copy_num=args.perturb_q_copy_num,
        perturb_std_dev=args.perturb_std_dev,
        perturb_q_learning_rate=args.perturb_q_learning_rate,
        perturb_grad_step=args.perturb_grad_step,
        q_grad_st_after_init=args.q_grad_st_after_init,
        ###
        weight_initialize=args.weight_initialize,
        w_schedule_option=args.w_schedule_option,
        ##
        stats_window_size=args.stats_window_size,
        ## alpha scheduling for SQL variants
        ent_alpha_act_init=args.ent_alpha_act_init,
        annealing_step=args.annealing_step,
        device='cpu' # we used cpu device as default
    )
    model.learn(total_timesteps=args.total_timesteps,
                tb_log_name="MaxminMFQ")

    wandb.finish()