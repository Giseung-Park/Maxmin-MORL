import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MultiInputPolicy, QNetwork, SQLPolicy, SoftQNetwork
import math
# from multiprocessing import Process, Queue, cpu_count
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate

SelfDQN = TypeVar("SelfDQN", bound="DQN")

import pdb

class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "DQNPolicy": DQNPolicy,
        "SQLPolicy": SQLPolicy, ## Newly added for our algorithm
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: Union[QNetwork, SoftQNetwork]
    q_net_target: Union[QNetwork, SoftQNetwork]
    policy: Union[DQNPolicy, SQLPolicy]

    def __init__(
        self,
        # policy: Union[str, Type[DQNPolicy]],
        policy: Union[str, Type[ Union[DQNPolicy, SQLPolicy] ]], ##
        env: Union[GymEnv, str], ##
        learning_rate: Union[float, Schedule] = 1e-4, ##
        buffer_size: int = 1_000_000, ##
        learning_starts: int = 50000, ##
        batch_size: int = 32,
        tau: float = 1.0, ##
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4, ##
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000, ##
        exploration_fraction: float = 0.1, ##
        exploration_initial_eps: float = 1.0, ##
        exploration_final_eps: float = 0.05, ##
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0, ##
        seed: Optional[int] = None, ##
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        r_dim: int = 1, ##
        r_dim_policy: int = 1, ##
        ent_alpha: float = None, # default for DQN
        weight_decay: float = 0, ## default value in Adam
        double_q: bool = False, ## default False. only applicalbe for baseline DQN
        explicit_w_input: bool = False, # For Maxmin DQN, this is not used. consider as dummy.
        scalarize: str = 'min', ## Only applicable for the Naive DQN baselines.
        ## alpha scheduling for SQL variants. For DQNs, consider as dummy.
        ent_alpha_act_init: Optional[float] = None,
        annealing_step: Optional[int] = None,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
            r_dim=r_dim,
            r_dim_policy=r_dim_policy,
            ent_alpha=ent_alpha,
            weight_decay=weight_decay,
            explicit_w_input=explicit_w_input,
            scalarize=scalarize,
            ent_alpha_act_init=ent_alpha_act_init,
            annealing_step=annealing_step,
        )
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        ### double q learning
        self.double_q = double_q

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target.forward(replay_data.next_observations) # [32,ac_dim] if r_dim_policy=1, [32,ac_dim,r_dim] if r_dim_policy > 1

                if self.r_dim_policy > 1:
                    multi_dim_reward = replay_data.rewards # [32, r_dim]
                    if self.double_q:
                        double_current_q = self.q_net.forward(replay_data.next_observations)
                        act_select_target = multi_dim_reward.unsqueeze(dim=1) + self.gamma * double_current_q # [32, 1, r_dim] + [32,ac_dim,r_dim] = [32,ac_dim,r_dim]
                    else:
                        act_select_target = multi_dim_reward.unsqueeze(dim=1) + self.gamma * next_q_values # [32, 1, r_dim] + [32,ac_dim,r_dim] = [32,ac_dim,r_dim]

                    if self.scalarize == 'min':
                        act_select_target_scal, _ = th.min(act_select_target, dim=-1) # scalarize: [32,ac_dim]
                    elif self.scalarize == 'mean': # utilitarian Naive DQN
                        act_select_target_scal = th.mean(act_select_target, dim=-1)  # scalarize: [32,ac_dim]
                    else:  # other scalar function
                        raise NotImplementedError

                    selected_act = th.argmax(act_select_target_scal, dim=1).unsqueeze(dim=-1) # [32] -> [32,1], index range: ac_space
                    tiled_selected_act = th.tile(selected_act, (1,self.r_dim_policy)).unsqueeze(dim=1) # [32,8] -> [32,1,8]
                    selected_target = th.gather(next_q_values, dim=1, index=tiled_selected_act).squeeze(dim=1) # [32, 1, r_dim] -> [32, r_dim]

                    # replay_data.rewards: [32, rew_dim], replay_data.dones: [32,1]
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * selected_target

                elif self.r_dim_policy == 1: ## Utilitarian DQN
                    if self.double_q:
                        double_current_q = self.q_net.forward(replay_data.next_observations) # [32,ac_dim] if r_dim_policy=1
                        _, double_current_q_idx = double_current_q.max(dim=1)  # [32] if r_dim=1,
                        selected_q_values = th.gather(next_q_values, dim=1, index=double_current_q_idx.unsqueeze(dim=1))  # [32,1] or  [32, 1, r_dim]
                    else:
                        # Follow greedy policy: use the one with the highest value
                        selected_q_values, _ = next_q_values.max(dim=1) # [32] if r_dim=1,
                        # Avoid potential broadcast issue
                        selected_q_values = selected_q_values.reshape(-1, 1) # [32,1] if r_dim=1,
                    # 1-step TD target
                    # replay_data.rewards: [32, rew_dim], replay_data.dones: [32,1]
                    #### Take mean directly from reward
                    target_q_values = th.mean(replay_data.rewards, dim=-1, keepdim=True) + (1 - replay_data.dones) * self.gamma * selected_q_values
                else:
                    raise NotImplementedError

            # Get current Q-values estimates
            current_q_values = self.q_net.forward(replay_data.observations) # [32,ac_dim] if r_dim=1, [32,ac_dim,r_dim] if r_dim > 1

            # Retrieve the q-values for the actions from the replay buffer
            idx = replay_data.actions.long() # replay_data.actions.long(): [32,1]
            if self.r_dim_policy > 1:
                idx = th.tile(idx, (1, self.r_dim_policy)).unsqueeze(dim=1) # [32,r_dim] -> [32,1,r_dim]

            current_q_values = th.gather(current_q_values, dim=1, index=idx) # [32,1] or  [32, 1, r_dim]
            if self.r_dim_policy > 1:
                current_q_values = th.squeeze(current_q_values, dim=1)
            assert len(current_q_values.shape) == 2 # [32,r_dim]

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        scalarize: str = 'min',
        timesteps: int = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """

        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic,
                                                scalarize=scalarize, timesteps=timesteps)
        return action, state

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

class Weight(th.nn.Module):
    def __init__(self,
                 r_dim: int=2,
                 initialize: Union[str, List[float]]='uniform',
                 device: str = 'auto'
        ):
        super(Weight, self).__init__()

        self.device = device

        # Declare a trainable parameter
        if initialize == 'uniform':
            self.weight = th.nn.Parameter(th.full((r_dim,1), 1/r_dim, dtype=th.float32, device=self.device))  # Example: a 1D tensor with 1/r_dim values
            # print("Weight", self.weight)
        elif initialize == 'dirichlet':
            # random sampling
            dirichlet_distribution = th.distributions.dirichlet.Dirichlet(th.ones(r_dim, dtype=th.float32)) # flat
            samples = dirichlet_distribution.sample().to(self.device) #[r_dim]
            self.weight = th.nn.Parameter(samples.unsqueeze(dim=-1))
            # print("Weight", self.weight)
        elif isinstance(initialize, list):
            self.weight = th.nn.Parameter(th.tensor(initialize, dtype=th.float32, device=self.device).unsqueeze(dim=-1))
            # print("Weight", self.weight)
        else:
            raise NotImplementedError

        self.matrix = th.zeros(r_dim, r_dim, device=self.device)
        for i in range(r_dim):
            for j in range(i + 1):
                self.matrix[i, j] = 1.0 / (i + 1)
        self.intercept = self.matrix[:,0].unsqueeze(dim=-1)  # [r_dim, 1]

        self.zero_vector = th.zeros(r_dim, device=self.device)

    def step(self, lr, grad):
        # Update the weights using the gradient and learning rate
        with th.no_grad():
            naive_weight = self.weight.data.squeeze(dim=-1) - lr * grad # minus because we conduct gradient descent. [r_dim, ]

            ## Now we calculate projection onto unit simplex
            sorted_weight, _ = th.sort(naive_weight, descending=True) # [r_dim, ]
            sorted_weight = sorted_weight.unsqueeze(dim=-1) # [r_dim, 1]

            criterion = sorted_weight - th.matmul(self.matrix, sorted_weight) + self.intercept # [r_dim, 1]
            threshold_idx = th.sum(criterion > 0).item() # range 1 to r_dim
            lmbda = criterion[threshold_idx-1] - sorted_weight[threshold_idx-1] # [1,]

            ### Final result
            self.weight.data = th.max(naive_weight + lmbda, self.zero_vector).unsqueeze(dim=-1) # [r_dim, ] -> [r_dim, 1]

    def forward(self, input):
        input = input.to(self.device)
        # Use the trainable parameter in the forward pass
        assert len(input.shape) == 2
        output = th.matmul(input, self.weight)
        return output


class MaxminMFQ(DQN):
    """
    Maxmin Model-free Q-learning (MaxminMFQ)
    Default hyperparameters are taken from the DQN paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """
    def __init__(
        self,
        # policy: Union[str, Type[DQNPolicy]], ###$
        policy: Union[str, Type[Union[DQNPolicy, SQLPolicy]]],
        env: Union[GymEnv, str], ###$
        learning_rate: Union[float, Schedule] = 1e-4, ###$
        buffer_size: int = 1_000_000,  ###$
        learning_starts: int = 50000, ###$
        train_freq: Union[int, Tuple[int, str]] = 4, ###$
        target_update_interval: int = 10000, ###$
        tau: float = 0.001, ###$
        exploration_initial_eps: float = 1.0, ###$
        exploration_final_eps: float = 0.05, ###$
        exploration_fraction: float = 0.1, ###$
        verbose: int = 0, ###$
        seed: Optional[int] = None, ###$
        r_dim: int = 1, ###$
        r_dim_policy: int = 1, ###$
        ent_alpha: float = 0.1, ###$
        weight_decay: float = 0,  ###$ default is 0
        ########### perturbation parameters
        #### w update threshold
        soft_q_init_fraction: float = 0.1, #$ 0.1
        #### w update params
        perturb_w_learning_rate: Union[float, Schedule] = 1e-4, #$ Maybe require Scheduler
        #### w grad calculation $
        period_cal_w_grad: int = 1,
        perturb_q_copy_num: int = 20,  # N_p > r_dim + 1
        perturb_std_dev: float = 0.01,
        ### perturb q update
        perturb_q_learning_rate: Union[float, Schedule] = 1e-4, #  Put in Adam optimizer. We think constant lr is OK.
        perturb_grad_step: int = 1,
        perturb_q_batch_size: int = 32, # set as 32
        ### Main Q update after initialization phase - gradient steps
        q_grad_st_after_init: int = 1,  ###
        explicit_w_input: bool = False,
        weight_initialize: Union[str, List[float]]='uniform',
        # period_main_grad: int=1,
        w_schedule_option: str = 'sqrt_inverse',
        stats_window_size: int = 100,
        ## alpha scheduling for SQL variants
        ent_alpha_act_init: float = 0.5,
        annealing_step: int = 10000,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__( # Call the __init__ method of the parent class
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            train_freq=train_freq,
            target_update_interval=target_update_interval,
            tau=tau,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_fraction=exploration_fraction,
            verbose=verbose,
            seed=seed,
            r_dim=r_dim,
            r_dim_policy=r_dim_policy,
            ent_alpha=ent_alpha,
            weight_decay=weight_decay,
            explicit_w_input=explicit_w_input,
            stats_window_size=stats_window_size,
            ent_alpha_act_init=ent_alpha_act_init,
            annealing_step=annealing_step,
            device=device
        )

        ### Add new parameters: weight
        assert perturb_q_copy_num >= r_dim + 1 ## For linear regression. np.linalg.det(X.T @ X) deviates from 0 if perturb_q_copy_num increases

        # self.weight = Weight(r_dim=r_dim)
        self.weight = Weight(r_dim=r_dim, initialize=weight_initialize, device=self.device)
        self.ent_alpha = ent_alpha

        self.init_state_tensor = th.tensor([1.] + [0. for _ in range(36)]).unsqueeze(0).to(self.device)

        ### Params for perturbation
        self.soft_q_init_fraction = soft_q_init_fraction
        self.perturb_std_dev = perturb_std_dev         ### std dev for Gaussian noise
        self.perturb_q_copy_num = perturb_q_copy_num
        self.sqrt_q_copy_num = round(math.sqrt(self.perturb_q_copy_num),2)
        self.period_cal_w_grad = period_cal_w_grad ### Period for gradient calculation

        self.perturb_q_batch_size = perturb_q_batch_size
        self.perturb_w_learning_rate = perturb_w_learning_rate
        self.perturb_q_learning_rate = perturb_q_learning_rate
        self.perturb_grad_step = perturb_grad_step
        self.q_grad_st_after_init = q_grad_st_after_init

        ### w_lr, perturb_q_lr scheduling - function
        # self.w_lr_schedule = get_schedule_fn(self.perturb_w_learning_rate)
        self.perturb_q_lr_schedule = get_schedule_fn(self.perturb_q_learning_rate)

        self.perturb_q_net_list = [self.policy.make_q_net() for _ in range(self.perturb_q_copy_num)]

        self.combined_parameters = []
        for perturb_q_net in self.perturb_q_net_list:
            self.combined_parameters.extend(perturb_q_net.parameters())
        self.perturb_q_optimizer = th.optim.Adam(self.combined_parameters, lr=self.perturb_q_lr_schedule(1))

        for perturb_q_net in self.perturb_q_net_list:
            perturb_q_net.set_training_mode(False) # Temporary

        ### Target Q
        self.perturb_q_net_target = self.policy.make_q_net()
        self.perturb_q_net_target.set_training_mode(False) # Permanent

        ### Calculated gradient of w
        self.w_grad = th.zeros(r_dim, dtype=th.float32)

        # ## w scheduling option
        self.w_schedule_option = w_schedule_option

    def update_perturb_q_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.perturb_q_lr_schedule(self._current_progress_remaining))

    ### Overriding train function. Here, we should use both (i) soft-q learning and (ii) weight update.
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if self.num_timesteps >= int(self.soft_q_init_fraction * self._total_timesteps):
            if self.num_timesteps%self.period_cal_w_grad == 0: ## calculate projected gradient of w

                ## Perturb weight
                perturbed_weight = self.weight.weight + th.randn(self.weight.weight.shape[0], self.perturb_q_copy_num, device=self.device) * self.perturb_std_dev  # [r_dim, q_copy_num]

                ## Make copies of Q_net
                for perturb_q_net in self.perturb_q_net_list:
                    perturb_q_net.set_training_mode(True)
                    perturb_q_net.load_state_dict(self.q_net.state_dict())

                ## Make a copy of Target Q_net
                self.perturb_q_net_target.load_state_dict(self.q_net_target.state_dict())

                # Schedule Optimizer
                self.update_perturb_q_learning_rate(self.perturb_q_optimizer)

                for _ in range(self.perturb_grad_step):
                    # Sample replay buffer - For now we only give perturbation w.r.t. weight, so we fix sampled buffer. (difference is from only perturbation)
                    replay_data = self.replay_buffer.sample(self.perturb_q_batch_size,env=self._vec_normalize_env)  # type: ignore[union-attr]

                    # target value calculation.
                    with th.no_grad():
                        # Compute the next Q-values using the target network
                        next_q_values = self.perturb_q_net_target.forward(replay_data.next_observations)  # [perturb_q_batch_size,ac_dim]
                        ## soft-q Target
                        next_q_values = self.ent_alpha * th.logsumexp(next_q_values / self.ent_alpha, dim=-1,keepdim=True)  # [perturb_q_batch_size,1]
                        # 1-step Soft TD target
                        # replay_data.rewards: [perturb_q_batch_size, r_dim], replay_data.dones: [perturb_q_batch_size,1]
                        # target_q_values: [perturb_q_batch_size, q_copy_num]
                        target_q_values = th.matmul(replay_data.rewards, perturbed_weight) + (1 - replay_data.dones) * self.gamma * next_q_values

                    ### Ver 3. Single process
                    current_q_values_p_list = []
                    for perturb_q_net in self.perturb_q_net_list:
                        # Get current Q-values estimates
                        current_q_values_p = perturb_q_net.forward(replay_data.observations)  # [perturb_q_batch_size,ac_dim]
                        current_q_values_p_list.append(current_q_values_p)

                    current_q_values_p_list = th.stack(current_q_values_p_list, dim=2) # [perturb_q_batch_size, ac_dim, q_copy_num]=[32,4,20]
                    # Retrieve the q-values for the actions from the replay buffer
                    idx = replay_data.actions.long().unsqueeze(-1)
                    idx = idx.repeat(1,1,self.perturb_q_copy_num) # [perturb_q_batch_size,1,self.perturb_q_copy_num]
                    current_q_values = th.gather(current_q_values_p_list, dim=1, index=idx).squeeze(dim=1)  # [perturb_q_batch_size, q_copy_num]

                    # Compute Huber loss (less sensitive to outliers)
                    loss = F.smooth_l1_loss(current_q_values, target_q_values)
                    # loss = F.mse_loss(current_q_values_p, target_q_values)

                    # Optimize the Q
                    self.perturb_q_optimizer.zero_grad()
                    loss.backward()
                    # Clip gradient norm
                    th.nn.utils.clip_grad_norm_(self.combined_parameters, self.sqrt_q_copy_num*self.max_grad_norm)
                    self.perturb_q_optimizer.step()

                with th.no_grad():
                    ## Calculate L(w+eps)
                    perturb_q_list = []
                    for perturb_q_net in self.perturb_q_net_list:
                        # Switch to eval mode (this affects batch norm / dropout)
                        perturb_q_net.set_training_mode(False) ## kind of "Discarding"
                        new_perturb_q = perturb_q_net.forward(self.init_state_tensor)  # [1,ac_dim]

                        new_perturb_q = self.ent_alpha * th.logsumexp(new_perturb_q / self.ent_alpha, dim=-1)  # [1,]
                        perturb_q_list.append(new_perturb_q.item())

                    ### Conduct linear regression
                    X = perturbed_weight.transpose(0, 1).to(self.device)  # [q_copy_num, r_dim]
                    y = th.tensor(perturb_q_list, dtype=th.float32, device=self.device) # [q_copy_num,]
                    # Add a column of ones for the intercept
                    ones_column = th.ones(X.size(0), dtype=X.dtype, device=self.device) # [q_copy_num,]
                    X = th.cat((ones_column.unsqueeze(1), X), dim=1) # [q_copy_num, r_dim+1]
                    try:
                        coefficients = th.linalg.inv(X.T @ X) @ X.T @ y # [r_dim+1,]
                    except:
                        coefficients = th.linalg.pinv(X.T @ X) @ X.T @ y

                    ### Update linear weight
                    self.w_grad = coefficients[1:] # [r_dim,]

                    del perturb_q_list, X, y

            ### Now we update parameter using proj_grad
            current_timestep_w = self.num_timesteps - int(self.soft_q_init_fraction*self._total_timesteps) + 1
            # self.weight.step(lr=self.perturb_w_learning_rate / math.sqrt(current_timestep_w), grad=self.w_grad)
            if self.w_schedule_option == 'sqrt_inverse':
                self.weight.step(lr=self.perturb_w_learning_rate / math.sqrt(current_timestep_w), grad=self.w_grad)
            elif self.w_schedule_option == 'inverse':
                self.weight.step(lr=self.perturb_w_learning_rate / current_timestep_w, grad=self.w_grad)
            elif self.w_schedule_option == 'linear': # for now, we use 0.1 end ratio
                max_step_w = int( (1-self.soft_q_init_fraction)*self._total_timesteps )
                self.weight.step(lr=self.perturb_w_learning_rate *
                                    ( 1 + (0.1-1)*(current_timestep_w-1)/max_step_w   ), grad=self.w_grad)
            else:
                raise NotImplementedError

            # self.weight.step(lr=self.w_lr_schedule(self._current_progress_remaining), grad=self.w_grad)

            ### After init phase, we increase gradient steps
            gradient_steps = self.q_grad_st_after_init


        #### Main Q update
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        ### Main Q update begins
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size,env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target.forward(replay_data.next_observations)  # [32,ac_dim] if r_dim_policy=1 (not r_dim)

                ## soft-q
                next_q_values = self.ent_alpha * th.logsumexp(next_q_values / self.ent_alpha, dim=-1,
                                                              keepdim=True)  # [32,1]

                # 1-step TD target
                # target_q_values: [32,1] // replay_data.rewards: [32, r_dim], replay_data.dones: [32,1]
                target_q_values = self.weight.forward(replay_data.rewards) + (
                            1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net.forward(
                replay_data.observations)  # [32*4,ac_dim] if r_dim_policy=1 (not r_dim)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())  # [32*4,1]
            assert len(current_q_values.shape) == 2  # [32,1]

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            # loss = F.mse_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        ### Main Q update ends

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))