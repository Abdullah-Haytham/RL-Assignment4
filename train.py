"""Train continuous-control RL agents on Gymnasium environments with WandB logging."""

import argparse
import os
from collections import deque
from typing import Any, Dict, List, Optional

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics, TransformObservation
try:
    from gymnasium.wrappers import FrameStack  # gymnasium>=0.28
except ImportError:  # pragma: no cover - fallback for older versions
    FrameStack = None
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import wandb

from rl.agents import (
    ContinuousPPOAgent,
    ContinuousSACAgent,
    TD3Agent,
)


CONTINUOUS_ENV_KWARGS: Dict[str, Dict[str, object]] = {
    "LunarLander-v3": {"continuous": True},
    "CarRacing-v3": {"continuous": True},
}


def flatten_obs(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class SimpleFrameStack(gym.Wrapper):
    """Fallback frame stacker for environments when Gymnasium's FrameStack is unavailable."""

    def __init__(self, env: gym.Env, num_stack: int) -> None:
        super().__init__(env)
        if num_stack <= 0:
            raise ValueError("num_stack must be positive")
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Box):
            raise ValueError("SimpleFrameStack only supports Box observation spaces")

        low = np.repeat(obs_space.low, num_stack, axis=-1)
        high = np.repeat(obs_space.high, num_stack, axis=-1)
        self.observation_space = spaces.Box(low=low, high=high, dtype=obs_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(np.asarray(obs, dtype=np.float32))
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(np.asarray(obs, dtype=np.float32))
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        if len(self.frames) < self.num_stack:
            first = self.frames[0] if self.frames else np.zeros_like(self.env.observation_space.low)
            while len(self.frames) < self.num_stack:
                self.frames.appendleft(first)
        return np.concatenate(list(self.frames), axis=-1)


FrameStackWrapper = FrameStack if FrameStack is not None else SimpleFrameStack


def make_env(env_name, seed=None, record_video=False, video_folder="videos", algo=None, video_frequency=50):
    render_mode = 'rgb_array' if record_video else None
    env_kwargs = CONTINUOUS_ENV_KWARGS.get(env_name, {}).copy()
    env = gym.make(env_name, render_mode=render_mode, **env_kwargs)

    if record_video:
        video_path = os.path.join(video_folder, algo.upper() if algo else "", env_name)
        os.makedirs(video_path, exist_ok=True)

        def should_record(episode_id):
            return episode_id == 0 or ((episode_id + 1) % video_frequency == 0)

        env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=should_record, name_prefix=env_name)

    if env_name.startswith("CarRacing"):
        env = RecordEpisodeStatistics(env)

        grayscale_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        base_shape = env.observation_space.shape
        if len(base_shape) != 3:
            raise ValueError("CarRacing environment expected 3D observations")

        gray_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(base_shape[0], base_shape[1], 1),
            dtype=np.float32,
        )

        def to_grayscale(obs):
            obs_array = np.asarray(obs, dtype=np.float32)
            gray = np.tensordot(obs_array[..., :3], grayscale_weights, axes=(-1, 0))
            gray = np.expand_dims(gray, axis=-1)
            return gray / 255.0

        env = TransformObservation(env, to_grayscale, observation_space=gray_space)
        env = FrameStackWrapper(env, 4)

    if seed is not None:
        try:
            env.reset(seed=seed)
            if hasattr(env.action_space, 'seed'):
                env.action_space.seed(seed)
            if hasattr(env.observation_space, 'seed'):
                env.observation_space.seed(seed)
        except Exception:
            pass
    return env


class CarRacingPPOAgent:
    """PPO agent with CNN encoder tailored for CarRacing observations."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        k_epochs: int = 10,
        entropy_coef: float = 0.0,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        log_std_bounds: Optional[tuple[float, float]] = (-5.0, 2.0),
    ) -> None:
        if len(observation_space.shape) != 3:
            raise ValueError("CarRacing agent expects 3D observations (H, W, C or C, H, W)")

        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.log_std_bounds = log_std_bounds or (-5.0, 2.0)

        self.channels_last = observation_space.shape[-1] <= 12
        if self.channels_last:
            self.height, self.width, self.in_channels = observation_space.shape
        else:
            self.in_channels, self.height, self.width = observation_space.shape

        action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0
        self.action_dim = action_space.shape[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        ).to(device)

        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.height, self.width, device=device)
            encoder_out_dim = self.encoder(dummy).shape[1]

        self.feature_layer = nn.Sequential(
            nn.Linear(encoder_out_dim, 512),
            nn.ReLU(),
        ).to(device)

        self.actor_mean = nn.Linear(512, self.action_dim).to(device)
        self.actor_log_std = nn.Parameter(torch.zeros(self.action_dim, device=device))
        self.critic = nn.Linear(512, 1).to(device)

        parameters = (
            list(self.encoder.parameters())
            + list(self.feature_layer.parameters())
            + list(self.actor_mean.parameters())
            + [self.actor_log_std]
            + list(self.critic.parameters())
        )
        self.optimizer = torch.optim.Adam(parameters, lr=lr)
        self._optim_params = parameters

        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[torch.Tensor] = []
        self.uses_replay_buffer = False

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scale + self.action_bias

    def _obs_to_tensor(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            tensor = obs.float().to(self.device)
            if tensor.dim() == 3:
                if self.channels_last:
                    tensor = tensor.permute(2, 0, 1)
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 4 and self.channels_last:
                tensor = tensor.permute(0, 3, 1, 2)
            return tensor

        obs_array = np.asarray(obs, dtype=np.float32)
        if obs_array.ndim != 3:
            raise ValueError("Expected 3D observation for CarRacing agent")
        if self.channels_last:
            obs_array = np.transpose(obs_array, (2, 0, 1))
        tensor = torch.from_numpy(obs_array).unsqueeze(0).to(self.device)
        return tensor

    def _forward_features(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs_tensor)
        return self.feature_layer(features)

    def _distribution(self, features: torch.Tensor) -> Normal:
        mean = self.actor_mean(features)
        min_log_std, max_log_std = self.log_std_bounds
        log_std = torch.clamp(self.actor_log_std, min_log_std, max_log_std)
        log_std = log_std.expand_as(mean)
        std = log_std.exp()
        return Normal(mean, std)

    def _value_from_tensor(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(obs_tensor)
        return self.critic(features).squeeze(-1)

    def select_action(self, state, deterministic: bool = False) -> np.ndarray:
        obs_tensor = self._obs_to_tensor(state)
        features = self._forward_features(obs_tensor)
        dist = self._distribution(features)

        if deterministic:
            pre_tanh = dist.mean
            log_prob = None
        else:
            pre_tanh = dist.rsample()
            log_prob = dist.log_prob(pre_tanh)

        action = torch.tanh(pre_tanh)
        scaled_action = self._scale_action(action)
        value = self._value_from_tensor(obs_tensor)

        if not deterministic and log_prob is not None:
            corrected_log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
            corrected_log_prob = corrected_log_prob.sum(dim=-1, keepdim=True)
            self.states.append(obs_tensor.detach())
            self.actions.append(action.detach())
            self.log_probs.append(corrected_log_prob.detach())
            self.values.append(value.detach())

        return scaled_action.squeeze(0).detach().cpu().numpy()

    def push_transition(self, state, action, next_state, reward: float, done: bool = False) -> None:
        _ = state
        _ = action
        _ = next_state
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    def _evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._forward_features(states)
        dist = self._distribution(features)

        actions_clamped = actions.clamp(-0.999, 0.999)
        pre_tanh = torch.atanh(actions_clamped)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1 - actions_clamped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def optimize(self, last_state: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        if not self.states:
            return None

        states = torch.cat(self.states, dim=0).to(self.device)
        actions = torch.cat(self.actions, dim=0).to(self.device)
        old_log_probs = torch.cat(self.log_probs, dim=0).to(self.device)
        values = torch.cat(self.values, dim=0).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)

        if last_state is not None:
            last_tensor = self._obs_to_tensor(last_state)
            next_value = self._value_from_tensor(last_tensor).detach()
        else:
            next_value = torch.zeros(1, device=self.device)

        returns: List[float] = []
        advantages: List[float] = []
        gae = 0.0
        values_with_last = torch.cat([values, next_value])
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values_with_last[step + 1] * (1 - dones[step])
                - values_with_last[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_with_last[step])

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = (
            (advantages_tensor - advantages_tensor.mean())
            / (advantages_tensor.std(unbiased=False) + 1e-8)
        )

        metrics: Dict[str, Any] = {}
        for _ in range(self.k_epochs):
            log_probs, entropy = self._evaluate_actions(states, actions)
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages_tensor.unsqueeze(-1)
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_tensor.unsqueeze(-1)
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = self._value_from_tensor(states)
            value_loss = F.mse_loss(value_pred, returns_tensor)

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._optim_params, self.max_grad_norm)
            self.optimizer.step()

            metrics = {
                'loss': float(loss.item()),
                'policy_loss': float(policy_loss.item()),
                'value_loss': float(value_loss.item()),
                'entropy': float(entropy.mean().item()),
                'advantages_mean': float(advantages_tensor.mean().item()),
                'advantages_std': float(advantages_tensor.std(unbiased=False).item()),
            }

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        return metrics

    def save(self, path: str) -> None:
        torch.save(
            {
                'encoder': self.encoder.state_dict(),
                'feature_layer': self.feature_layer.state_dict(),
                'actor_mean': self.actor_mean.state_dict(),
                'actor_log_std': self.actor_log_std.detach().cpu(),
                'critic': self.critic.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.feature_layer.load_state_dict(checkpoint['feature_layer'])
        self.actor_mean.load_state_dict(checkpoint['actor_mean'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'actor_log_std' in checkpoint:
            self.actor_log_std.data.copy_(checkpoint['actor_log_std'].to(self.device))


def train_carracing_ppo(
    env,
    agent: CarRacingPPOAgent,
    run,
    env_name: str,
    algo: str,
    episodes: int,
    save_best: bool,
) -> None:
    print(f"Training {algo.upper()} on {env_name} for {episodes} episodes...")
    print(
        f"Device: {agent.device.type}, Action dim: {env.action_space.shape[0]}, Observations: {env.observation_space.shape}"
    )

    best_reward = -float('inf')
    for i_episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.push_transition(obs, action, next_obs, float(reward), done)

            obs = next_obs
            total_reward += float(reward)
            steps += 1

            if done:
                bootstrap_state = next_obs if truncated else None
                metrics = agent.optimize(last_state=bootstrap_state)

                log_dict = {
                    'train/episode_reward': total_reward,
                    'train/episode_length': steps,
                    'train/episode': i_episode,
                }
                if metrics is not None:
                    for key, value in metrics.items():
                        log_dict[f'train/{key}'] = value
                run.log(log_dict)

                if (i_episode + 1) % 10 == 0:
                    loss_str = f"{metrics['loss']:.3f}" if metrics else "N/A"
                    entropy_str = f"{metrics['entropy']:.3f}" if metrics else "N/A"
                    print(
                        f"Episode {i_episode + 1}/{episodes} - Reward: {total_reward:.1f}, Length: {steps}, "
                        f"Loss: {loss_str}, Entropy: {entropy_str}"
                    )
                break

        if (i_episode + 1) % 100 == 0:
            os.makedirs('models', exist_ok=True)
            model_path = f"models/{env_name}_{algo}_ep{i_episode+1}.pt"
            agent.save(model_path)

        if save_best and total_reward > best_reward:
            best_reward = total_reward
            os.makedirs('models', exist_ok=True)
            best_path = f"models/{env_name}_{algo}_best.pt"
            agent.save(best_path)

    print("\nTraining completed!")
    os.makedirs('models', exist_ok=True)
    final_path = f"models/{env_name}_{algo}_final.pt"
    agent.save(final_path)
    print(f"Model saved to {final_path}")

    run.finish()
    env.close()


def train(
    env_name: str,
    algo: str,
    episodes: int = 1500,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    record_video: bool = True,
    video_frequency: int = 50,
    device: Optional[str] = None,
    project: str = "rl-ass4",
    entity: Optional[str] = None,
    seed: int = 42,
    sac_alpha: float = 0.2,
    sac_tau: float = 0.005,
    sac_replay_size: int = 1_000_000,
    sac_batch_size: int = 256,
    sac_warmup_steps: int = 1000,
    sac_updates_per_optimize: int = 1,
    sac_auto_entropy: bool = True,
    td3_policy_noise: float = 0.2,
    td3_noise_clip: float = 0.5,
    td3_policy_delay: int = 2,
    td3_exploration_noise: float = 0.1,
    ppo_clip: float = 0.2,
    ppo_epochs: int = 10,
    ppo_gae_lambda: float = 0.95,
    save_best: bool = True,
):
    device_name = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device_name)

    torch.manual_seed(seed)
    np.random.seed(seed)

    run_name = f"{algo}_{env_name}_{episodes}eps"
    config = {
        'env': env_name,
        'algo': algo,
        'episodes': episodes,
        'lr': lr,
        'gamma': gamma,
        'entropy_coef': entropy_coef,
        'value_coef': value_coef,
        'max_grad_norm': max_grad_norm,
        'seed': seed,
        'sac_alpha': sac_alpha,
        'sac_tau': sac_tau,
        'sac_replay_size': sac_replay_size,
        'sac_batch_size': sac_batch_size,
        'sac_warmup_steps': sac_warmup_steps,
        'sac_updates_per_optimize': sac_updates_per_optimize,
        'sac_auto_entropy': sac_auto_entropy,
        'td3_policy_noise': td3_policy_noise,
        'td3_noise_clip': td3_noise_clip,
        'td3_policy_delay': td3_policy_delay,
        'td3_exploration_noise': td3_exploration_noise,
        'ppo_clip': ppo_clip,
        'ppo_epochs': ppo_epochs,
        'ppo_gae_lambda': ppo_gae_lambda,
        'record_video': record_video,
        'video_frequency': video_frequency,
    }

    if env_name.startswith("CarRacing"):
        config.update({
            'frame_stack': 4,
            'observation_processing': 'grayscale_norm',
        })

    run = wandb.init(project=project, entity=entity, name=run_name, config=config)

    algo_lower = algo.lower()
    env = make_env(env_name, seed=seed, record_video=record_video, algo=algo, video_frequency=video_frequency)
    action_space = env.action_space
    if not isinstance(action_space, spaces.Box):
        raise ValueError("Continuous training script expects a continuous (Box) action space")

    if env_name.startswith("CarRacing") and algo_lower == "ppo":
        agent = CarRacingPPOAgent(
            observation_space=env.observation_space,
            action_space=action_space,
            device=torch_device,
            lr=lr,
            gamma=gamma,
            gae_lambda=ppo_gae_lambda,
            eps_clip=ppo_clip,
            k_epochs=ppo_epochs,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
        )

        train_carracing_ppo(
            env=env,
            agent=agent,
            run=run,
            env_name=env_name,
            algo=algo_lower,
            episodes=episodes,
            save_best=save_best,
        )
        return

    obs, _ = env.reset(seed=seed)
    obs_vector = flatten_obs(obs)
    n_observations = obs_vector.shape[0]

    if algo_lower == "sac":
        agent = ContinuousSACAgent(
            n_observations,
            action_space,
            device=torch_device,
            lr=lr,
            gamma=gamma,
            tau=sac_tau,
            alpha=sac_alpha,
            automatic_entropy_tuning=sac_auto_entropy,
            replay_size=sac_replay_size,
            batch_size=sac_batch_size,
            warmup_steps=sac_warmup_steps,
            updates_per_optimize=sac_updates_per_optimize,
        )
    elif algo_lower == "ppo":
        agent = ContinuousPPOAgent(
            n_observations,
            action_space,
            device=torch_device,
            lr=lr,
            gamma=gamma,
            gae_lambda=ppo_gae_lambda,
            eps_clip=ppo_clip,
            k_epochs=ppo_epochs,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
        )
    elif algo_lower == "td3":
        agent = TD3Agent(
            n_observations,
            action_space,
            device=torch_device,
            lr=lr,
            gamma=gamma,
            tau=sac_tau,
            policy_noise=td3_policy_noise,
            noise_clip=td3_noise_clip,
            policy_delay=td3_policy_delay,
            exploration_noise=td3_exploration_noise,
            replay_size=sac_replay_size,
            batch_size=sac_batch_size,
            warmup_steps=sac_warmup_steps,
            updates_per_optimize=sac_updates_per_optimize,
        )
    else:
        raise ValueError(f"Unsupported continuous algorithm {algo}")

    print(f"Training {algo.upper()} on {env_name} for {episodes} episodes...")
    print(f"Device: {torch_device.type}, Action dim: {action_space.shape[0]}, Observations: {n_observations}")

    best_reward = -float('inf')
    for i_episode in range(episodes):
        obs, _ = env.reset()
        obs_vec = flatten_obs(obs)
        state = torch.tensor(obs_vec, dtype=torch.float32, device=torch_device).unsqueeze(0)
        total_reward = 0.0
        t = 0
        last_metrics = None
        current_obs = obs_vec

        while True:
            if getattr(agent, 'uses_replay_buffer', False):
                action = agent.select_action(current_obs, deterministic=False)
            else:
                action = agent.select_action(state, deterministic=False)

            env_action = action
            if isinstance(env_action, torch.Tensor):
                env_action = env_action.detach().cpu().numpy()
            env_action = np.asarray(env_action, dtype=np.float32)
            action_tensor = torch.tensor(env_action, dtype=torch.float32, device=torch_device)

            next_obs_raw, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated

            next_obs_vec = flatten_obs(next_obs_raw)
            next_state = torch.tensor(next_obs_vec, dtype=torch.float32, device=torch_device).unsqueeze(0)
            reward_value = float(reward)

            if getattr(agent, 'uses_replay_buffer', False):
                agent.push_transition(current_obs, env_action, next_obs_vec, reward_value, done)
                step_metrics = agent.optimize()
                if step_metrics is not None:
                    last_metrics = step_metrics
            else:
                agent.push_transition(state, action_tensor, next_state, reward_value)

            state = next_state
            current_obs = next_obs_vec
            total_reward += reward_value
            t += 1

            if done:
                if getattr(agent, 'uses_replay_buffer', False):
                    metrics = last_metrics
                else:
                    bootstrap_state = next_state if truncated else None
                    metrics = agent.optimize(last_state=bootstrap_state)

                log_dict = {
                    'train/episode_reward': total_reward,
                    'train/episode_length': t,
                    'train/episode': i_episode,
                }
                if metrics is not None:
                    for key, value in metrics.items():
                        log_dict[f'train/{key}'] = value
                run.log(log_dict)

                if (i_episode + 1) % 10 == 0:
                    loss_str = f"{metrics['loss']:.3f}" if metrics else "N/A"
                    entropy_str = f"{metrics['entropy']:.3f}" if metrics else "N/A"
                    print(
                        f"Episode {i_episode + 1}/{episodes} - Reward: {total_reward:.1f}, Length: {t}, "
                        f"Loss: {loss_str}, Entropy: {entropy_str}"
                    )
                break

        if (i_episode + 1) % 100 == 0:
            os.makedirs('models', exist_ok=True)
            model_path = f"models/{env_name}_{algo}_ep{i_episode+1}.pt"
            agent.save(model_path)
        if save_best and total_reward > best_reward:
            best_reward = total_reward
            os.makedirs('models', exist_ok=True)
            best_path = f"models/{env_name}_{algo}_best.pt"
            agent.save(best_path)

    print("\nTraining completed!")
    os.makedirs('models', exist_ok=True)
    model_path = f"models/{env_name}_{algo}_final.pt"
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    run.finish()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLander-v3')
    parser.add_argument('--algo', type=str, choices=['sac', 'ppo', 'td3'], default='sac')
    parser.add_argument('--episodes', type=int, default=1500)
    parser.add_argument('--learning-rate', '--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=0.0)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--record-video', dest='record_video', action='store_true', help='Enable video recording')
    parser.add_argument('--no-record-video', dest='record_video', action='store_false', help='Disable video recording')
    parser.add_argument('--video-frequency', type=int, default=50)
    parser.add_argument('--project', type=str, default='rl-ass4')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sac-alpha', type=float, default=0.2)
    parser.add_argument('--sac-tau', type=float, default=0.005)
    parser.add_argument('--sac-replay-size', type=int, default=1_000_000)
    parser.add_argument('--sac-batch-size', type=int, default=256)
    parser.add_argument('--sac-warmup-steps', type=int, default=1000)
    parser.add_argument('--sac-updates-per-optimize', type=int, default=1)
    parser.add_argument('--sac-auto-entropy', action='store_true')
    parser.add_argument('--td3-policy-noise', type=float, default=0.2)
    parser.add_argument('--td3-noise-clip', type=float, default=0.5)
    parser.add_argument('--td3-policy-delay', type=int, default=2)
    parser.add_argument('--td3-exploration-noise', type=float, default=0.1)
    parser.add_argument('--ppo-clip', type=float, default=0.2)
    parser.add_argument('--ppo-epochs', type=int, default=10)
    parser.add_argument('--ppo-gae-lambda', type=float, default=0.95)
    parser.add_argument('--save-best', dest='save_best', action='store_true', help='Save best-performing model checkpoint')
    parser.add_argument('--no-save-best', dest='save_best', action='store_false', help='Skip saving best-performing checkpoint')
    parser.set_defaults(record_video=True, save_best=True)
    args = parser.parse_args()

    train(
        env_name=args.env,
        algo=args.algo,
        episodes=args.episodes,
        lr=args.learning_rate,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        record_video=args.record_video,
        video_frequency=args.video_frequency,
        project=args.project,
        entity=args.entity,
        seed=args.seed,
        device=args.device,
        sac_alpha=args.sac_alpha,
        sac_tau=args.sac_tau,
        sac_replay_size=args.sac_replay_size,
        sac_batch_size=args.sac_batch_size,
        sac_warmup_steps=args.sac_warmup_steps,
        sac_updates_per_optimize=args.sac_updates_per_optimize,
        sac_auto_entropy=args.sac_auto_entropy,
        td3_policy_noise=args.td3_policy_noise,
        td3_noise_clip=args.td3_noise_clip,
        td3_policy_delay=args.td3_policy_delay,
        td3_exploration_noise=args.td3_exploration_noise,
        ppo_clip=args.ppo_clip,
        ppo_epochs=args.ppo_epochs,
        ppo_gae_lambda=args.ppo_gae_lambda,
        save_best=args.save_best,
    )
