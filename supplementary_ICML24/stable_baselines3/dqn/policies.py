from typing import Any, Dict, List, Optional, Type

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
import pdb

class QNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        r_dim_policy: int, ## Added for MORL
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.action_dim = int(self.action_space.n)  # number of actions
        self.r_dim_policy = r_dim_policy # Added for MORL
        q_net = create_mlp(self.features_dim, self.r_dim_policy*self.action_dim, self.net_arch, self.activation_fn) # Revised for MORL
        ### input_dim: self.features_dim, output_dim: action_dim*r_dim_policy
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # batch=1 for rollout
        # [batch, ac_dim] if r_dim_policy=1 / [batch, ac_dim, r_dim_policy] if r_dim_policy > 1
        if self.r_dim_policy > 1:
            return self.q_net(self.extract_features(obs, self.features_extractor)).view(-1, self.action_dim, self.r_dim_policy)
        elif self.r_dim_policy == 1:
            return self.q_net(self.extract_features(obs, self.features_extractor))
        else:
            raise NotImplementedError

    def _predict(self, observation: th.Tensor, deterministic: bool = True, scalarize: str = 'min', timesteps: int=None) -> th.Tensor: ## For now, we add scalarize function 'f=min'
        q_values = self.forward(observation)
        # Greedy action
        if self.r_dim_policy > 1: # [batch(=1), ac_dim, r_dim_policy]
            if scalarize == 'min': ##
                scalarized_q_values, _ = th.min(q_values, dim=-1) # [1, ac_dim]
                return scalarized_q_values.argmax(dim=1).reshape(-1) # [1] (scalar)
            elif scalarize == 'mean':
                scalarized_q_values = th.mean(q_values, dim=-1)  # [1, ac_dim]
                return scalarized_q_values.argmax(dim=1).reshape(-1)  # [1] (scalar)
            else: # other scalar function
                raise NotImplementedError
        elif self.r_dim_policy == 1: # q_values [1, ac_dim] -> [1] (scalar). #used for utilitarian DQN
            return q_values.argmax(dim=1).reshape(-1)
        else:
            raise NotImplementedError

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

class SoftQNetwork(QNetwork):
    """
    Action-Value (Q-Value) network for Soft Q-learning

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        r_dim_policy: int, ## Added for MORL
        ent_alpha: float, ## required
        ent_alpha_act_init: float, ## required
        annealing_step: int, ## required
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            r_dim_policy=r_dim_policy,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        self.ent_alpha_act_init = ent_alpha_act_init
        self.ent_alpha_act_min = ent_alpha # same as training alpha
        assert self.ent_alpha_act_init >= self.ent_alpha_act_min

        self.annealing_step = annealing_step

        ## initalize
        self.ent_alpha_act = ent_alpha_act_init

    def schedule_alpha(self, current_timestep):
        # Linear
        frac = max(0, 1 - current_timestep / self.annealing_step)
        self.ent_alpha_act = self.ent_alpha_act_min + frac * (self.ent_alpha_act_init - self.ent_alpha_act_min)

    def _predict(self, observation: th.Tensor, deterministic: bool = True, scalarize: str= None, timesteps: int=None) -> th.Tensor:

        ### Original version
        # schedule action alpha
        self.schedule_alpha(current_timestep=timesteps)

        q_values = self.forward(observation) # [1,4]=[1, ac_dim]
        soft_q_probs = nn.functional.softmax(q_values/self.ent_alpha_act, dim=1) # [1,4]
        sampled_action = th.distributions.Categorical(soft_q_probs).sample()

        return sampled_action


class DQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    q_net: QNetwork
    q_net_target: QNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        r_dim_policy: int,  ## Newly added for MORL
        ent_alpha: float,
        weight_decay: float,
        explicit_w_input: bool,  ### For SQLPolicy only
        r_dim: int,  ### For SQLPolicy only
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU, # nn.ReLU
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        ## alpha scheduling for SQL variants. For DQNs, consider as dummy.
        ent_alpha_act_init: Optional[float] = None,
        annealing_step: Optional[int] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "r_dim_policy": r_dim_policy,  ## Added for MORL
        }

        if ent_alpha is not None:
            self.net_args["ent_alpha"] = ent_alpha

        ## alpha scheduling for SQL
        if ent_alpha_act_init is not None:
            self.net_args["ent_alpha_act_init"] = ent_alpha_act_init
        if annealing_step is not None:
            self.net_args["annealing_step"] = annealing_step

        ##### For SQLPolicy only. For now, it is dummy.
        self.explicit_w_input = explicit_w_input
        self.r_dim = r_dim

        ### Adam optimizer
        self.weight_decay = weight_decay

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # print("q_net", self.q_net)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            weight_decay=self.weight_decay,
            **self.optimizer_kwargs, # {}
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)

        return QNetwork(**net_args).to(self.device)

    # def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor: ## Not explicitly used
    #     return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True, scalarize: str = 'min', timesteps: int=None) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic, scalarize=scalarize, timesteps=timesteps)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.training = mode

class SQLPolicy(DQNPolicy):
    """
    Policy class with Q-Value Net and target net for SQL

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    q_net: SoftQNetwork
    q_net_target: SoftQNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        r_dim_policy: int,  ## Newly added for MORL
        ent_alpha: float,
        weight_decay: float,
        explicit_w_input: bool, ### For SQLPolicy only
        r_dim: int, ### For SQLPolicy only
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU, # nn.ReLU
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        ## alpha scheduling for SQL variants. For DQNs, consider as dummy.
        ent_alpha_act_init: float = None,
        annealing_step: int = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            r_dim_policy,
            ent_alpha,
            weight_decay,
            explicit_w_input, ### For SQLPolicy only
            r_dim, ### For SQLPolicy only
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            ent_alpha_act_init=ent_alpha_act_init,
            annealing_step=annealing_step
        )

    def make_q_net(self) -> SoftQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None) ## Here, 'features_dim' is added.

        ## add preference(omega) only in this SQL case.
        ## Increase feature_dim here! 'features_dim': 37 + self.r_dim
        # if self.explicit_w_input:
        #     net_args['features_dim'] += self.r_dim

        return SoftQNetwork(**net_args).to(self.device)

# MlpPolicy = DQNPolicy
# SoftMlpPolicy = SQLPolicy

class CnnPolicy(DQNPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class MultiInputPolicy(DQNPolicy):
    """
    Policy class for DQN when using dict observations as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
