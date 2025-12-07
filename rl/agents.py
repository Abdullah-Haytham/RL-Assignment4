"""Continuous control agents separated from discrete implementations."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Deque, Union

from collections import deque
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

class ReplayBuffer:
    """Simple FIFO replay buffer for off-policy agents."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(
        self,
        state: Union[Sequence[float], np.ndarray, float],
        action: Union[Sequence[float], np.ndarray, float],
        reward: float,
        next_state: Union[Sequence[float], np.ndarray, float],
        done: bool,
    ) -> None:
        state_arr = np.asarray(state, dtype=np.float32)
        action_arr = np.asarray(action)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        self.buffer.append((state_arr, action_arr, float(reward), next_state_arr, float(done)))

    def sample(self, batch_size: int) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), np.asarray(rewards, dtype=np.float32), list(next_states), np.asarray(dones, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.buffer)


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: type = nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class TanhGaussianPolicy(nn.Module):
    """Gaussian policy with Tanh squashing to keep actions in [-1, 1]."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        log_std_bounds: Tuple[float, float] = (-20.0, 2.0),
    ) -> None:
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.net = _build_mlp(obs_dim, hidden_dims, 2 * action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_log_std = self.net(obs)
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        min_log_std, max_log_std = self.log_std_bounds
        log_std = torch.tanh(log_std)
        log_std = min_log_std + 0.5 * (log_std + 1) * (max_log_std - min_log_std)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class DoubleQNetwork(nn.Module):
    """Twin Q network head for continuous control algorithms."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        input_dim = obs_dim + action_dim
        self.q1 = _build_mlp(input_dim, hidden_dims, 1)
        self.q2 = _build_mlp(input_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa)


class ContinuousSACAgent:
    """Soft Actor-Critic agent for continuous action spaces."""

    def __init__(
        self,
        obs_dim: int,
        action_space,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        hidden_dims: Sequence[int] = (256, 256),
        replay_size: int = 1_000_000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        updates_per_optimize: int = 1,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_optimize = updates_per_optimize
        self.total_steps = 0
        self.total_updates = 0
        self.automatic_entropy_tuning = automatic_entropy_tuning

        action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0
        self.action_dim = action_space.shape[0]

        self.policy = TanhGaussianPolicy(obs_dim, self.action_dim, hidden_dims).to(device)
        self.q_network = DoubleQNetwork(obs_dim, self.action_dim, hidden_dims).to(device)
        self.q_target = DoubleQNetwork(obs_dim, self.action_dim, hidden_dims).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.alpha = alpha
        if self.automatic_entropy_tuning:
            if target_entropy is None:
                target_entropy = -float(self.action_dim)
            self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None

        self.replay_buffer = ReplayBuffer(replay_size)
        self.uses_replay_buffer = True

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scale + self.action_bias

    def select_action(self, state, deterministic: bool = False) -> np.ndarray:
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_tensor = state.to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if deterministic:
            _, _, mean_action = self.policy.sample(state_tensor)
            action = mean_action
        else:
            action, _, _ = self.policy.sample(state_tensor)
        action = self._scale_action(action)
        return action.squeeze(0).detach().cpu().numpy()

    def push_transition(self, state, action, next_state, reward, done: bool = False) -> None:
        state_arr = np.asarray(state, dtype=np.float32)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        action_arr = np.asarray(action, dtype=np.float32)
        self.replay_buffer.push(state_arr, action_arr, float(reward), next_state_arr, float(done))
        self.total_steps += 1

    def optimize(self, last_state: Optional[torch.Tensor] = None) -> Optional[Dict[str, Any]]:
        _ = last_state
        if len(self.replay_buffer) < self.batch_size or self.total_steps < self.warmup_steps:
            return None

        metrics: List[Dict[str, float]] = []
        for _ in range(self.updates_per_optimize):
            if len(self.replay_buffer) < self.batch_size:
                break
            states_np, actions_np, rewards_np, next_states_np, dones_np = self.replay_buffer.sample(self.batch_size)
            states = torch.tensor(np.stack(states_np), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.stack(actions_np), dtype=torch.float32, device=self.device)
            rewards = torch.tensor(rewards_np, dtype=torch.float32, device=self.device).unsqueeze(-1)
            next_states = torch.tensor(np.stack(next_states_np), dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones_np, dtype=torch.float32, device=self.device).unsqueeze(-1)

            with torch.no_grad():
                next_action, next_log_prob, _ = self.policy.sample(next_states)
                next_action = self._scale_action(next_action)
                target_q1, target_q2 = self.q_target(next_states, next_action)
                target_min = torch.min(target_q1, target_q2)
                curr_alpha = self.alpha if not self.automatic_entropy_tuning else self.log_alpha.exp()
                target = rewards + self.gamma * (1 - dones) * (target_min - curr_alpha * next_log_prob)

            current_q1, current_q2 = self.q_network(states, actions)
            q_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            new_action, log_prob, _ = self.policy.sample(states)
            new_action = self._scale_action(new_action)
            q1_new = self.q_network.q1_forward(states, new_action)
            curr_alpha = self.alpha if not self.automatic_entropy_tuning else self.log_alpha.exp()
            policy_loss = (curr_alpha * log_prob - q1_new).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            alpha_loss_value = 0.0
            entropy = -log_prob.mean().item()
            if self.automatic_entropy_tuning and self.alpha_optimizer is not None and self.log_alpha is not None:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha_loss_value = float(alpha_loss.item())
                self.alpha = float(self.log_alpha.exp().item())

            for target_param, param in zip(self.q_target.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.total_updates += 1
            metrics.append({
                'loss': float((q_loss + policy_loss).item()),
                'policy_loss': float(policy_loss.item()),
                'value_loss': float(q_loss.item()),
                'entropy': float(entropy),
                'alpha': float(self.alpha),
                'alpha_loss': alpha_loss_value,
                'replay_buffer_size': float(len(self.replay_buffer)),
            })

        if not metrics:
            return None
        aggregated = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0].keys()}
        return aggregated

    def save(self, path: str) -> None:
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q_state_dict': self.q_network.state_dict(),
            'target_q_state_dict': self.q_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha.detach().cpu().item() if self.log_alpha is not None else None,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.q_network.load_state_dict(ckpt['q_state_dict'])
        self.q_target.load_state_dict(ckpt['target_q_state_dict'])
        self.policy_optimizer.load_state_dict(ckpt['policy_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(ckpt['q_optimizer_state_dict'])
        if 'alpha' in ckpt and ckpt['alpha'] is not None:
            self.alpha = float(ckpt['alpha'])
        if self.log_alpha is not None and 'log_alpha' in ckpt and ckpt['log_alpha'] is not None:
            value = torch.tensor([ckpt['log_alpha']], device=self.device)
            self.log_alpha.data.copy_(value)


class ContinuousPPOAgent:
    """PPO with Gaussian policy for continuous control."""

    def __init__(
        self,
        obs_dim: int,
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
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        self.policy = TanhGaussianPolicy(obs_dim, action_space.shape[0], hidden_dims).to(device)
        self.critic = _build_mlp(obs_dim, hidden_dims, 1).to(device)
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.critic.parameters()), lr=lr)

        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[torch.Tensor] = []
        self.uses_replay_buffer = False

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scale + self.action_bias

    def select_action(self, state, deterministic: bool = False) -> np.ndarray:
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_tensor = state.to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        if deterministic:
            _, _, mean_action = self.policy.sample(state_tensor)
            action = mean_action
            log_prob = None
        else:
            action, log_prob, _ = self.policy.sample(state_tensor)

        scaled_action = self._scale_action(action)
        value = self.critic(state_tensor).squeeze(-1)

        if not deterministic and log_prob is not None:
            self.states.append(state_tensor.detach())
            self.actions.append(action.detach())
            self.log_probs.append(log_prob.detach())
            self.values.append(value.detach())

        return scaled_action.squeeze(0).detach().cpu().numpy()

    def push_transition(self, state, action, next_state, reward, done: bool = False) -> None:
        _ = state
        _ = action
        _ = next_state
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    def _evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.policy(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        actions_clamped = actions.clamp(-0.999, 0.999)
        z = torch.atanh(actions_clamped)
        log_prob = normal.log_prob(z) - torch.log(1 - actions_clamped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def optimize(self, last_state: Optional[torch.Tensor] = None) -> Optional[Dict[str, Any]]:
        if not self.states:
            return None

        states = torch.cat(self.states).to(self.device)
        actions = torch.cat(self.actions).to(self.device)
        old_log_probs = torch.cat(self.log_probs).to(self.device)
        values = torch.cat(self.values).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)

        if last_state is not None:
            if not isinstance(last_state, torch.Tensor):
                last_state = torch.tensor(last_state, dtype=torch.float32, device=self.device)
            if last_state.dim() == 1:
                last_state = last_state.unsqueeze(0)
            next_value = self.critic(last_state).detach().squeeze(-1)
        else:
            next_value = torch.zeros(1, device=self.device)

        returns = []
        advantages = []
        gae = 0.0
        values_with_last = torch.cat([values, next_value])
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values_with_last[step + 1] * (1 - dones[step]) - values_with_last[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_with_last[step])

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std(unbiased=False) + 1e-8)

        metrics = {}
        for _ in range(self.k_epochs):
            log_probs, entropy = self._evaluate_actions(states, actions)
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages_tensor.unsqueeze(-1)
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_tensor.unsqueeze(-1)
            policy_loss = -torch.min(surr1, surr2).mean()
            values_pred = self.critic(states).squeeze(-1)
            value_loss = F.mse_loss(values_pred, returns_tensor)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
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
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient agent."""

    def __init__(
        self,
        obs_dim: int,
        action_space,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        exploration_noise: float = 0.1,
        replay_size: int = 1_000_000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        updates_per_optimize: int = 1,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_optimize = updates_per_optimize
        self.total_steps = 0
        self.total_updates = 0

        action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0
        self.action_dim = action_space.shape[0]

        self.actor = _build_mlp(obs_dim, hidden_dims, self.action_dim).to(device)
        self.actor_target = _build_mlp(obs_dim, hidden_dims, self.action_dim).to(device)
        self.critic = DoubleQNetwork(obs_dim, self.action_dim, hidden_dims).to(device)
        self.critic_target = DoubleQNetwork(obs_dim, self.action_dim, hidden_dims).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(replay_size)
        self.uses_replay_buffer = True

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(action) * self.action_scale + self.action_bias

    def select_action(self, state, deterministic: bool = False) -> np.ndarray:
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_tensor = state.to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        action = self._scale_action(self.actor(state_tensor))
        if not deterministic:
            noise = torch.normal(mean=0.0, std=self.exploration_noise, size=action.shape, device=self.device)
            action = action + noise
        action = torch.max(torch.min(action, self.action_bias + self.action_scale), self.action_bias - self.action_scale)
        return action.squeeze(0).detach().cpu().numpy()

    def push_transition(self, state, action, next_state, reward, done: bool = False) -> None:
        self.replay_buffer.push(
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        )
        self.total_steps += 1

    def optimize(self, last_state: Optional[torch.Tensor] = None) -> Optional[Dict[str, Any]]:
        _ = last_state
        if len(self.replay_buffer) < self.batch_size or self.total_steps < self.warmup_steps:
            return None

        metrics: List[Dict[str, float]] = []
        for _ in range(self.updates_per_optimize):
            states_np, actions_np, rewards_np, next_states_np, dones_np = self.replay_buffer.sample(self.batch_size)
            states = torch.tensor(np.stack(states_np), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.stack(actions_np), dtype=torch.float32, device=self.device)
            rewards = torch.tensor(rewards_np, dtype=torch.float32, device=self.device).unsqueeze(-1)
            next_states = torch.tensor(np.stack(next_states_np), dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones_np, dtype=torch.float32, device=self.device).unsqueeze(-1)

            with torch.no_grad():
                noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = self._scale_action(self.actor_target(next_states)) + noise
                next_action = torch.max(torch.min(next_action, self.action_bias + self.action_scale), self.action_bias - self.action_scale)
                target_q1, target_q2 = self.critic_target(next_states, next_action)
                target_q = torch.min(target_q1, target_q2)
                target = rewards + self.gamma * (1 - dones) * target_q

            current_q1, current_q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            update_actor = self.total_updates % self.policy_delay == 0
            actor_loss_value = 0.0
            entropy = 0.0
            if update_actor:
                actor_action = self._scale_action(self.actor(states))
                actor_loss = -self.critic.q1_forward(states, actor_action).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_loss_value = float(actor_loss.item())

                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.total_updates += 1
            metrics.append({
                'loss': float(critic_loss.item() + actor_loss_value),
                'policy_loss': actor_loss_value,
                'value_loss': float(critic_loss.item()),
                'entropy': float(entropy),
                'replay_buffer_size': float(len(self.replay_buffer)),
            })

        if not metrics:
            return None
        aggregated = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0].keys()}
        return aggregated

    def save(self, path: str) -> None:
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.actor_target.load_state_dict(ckpt['actor_target_state_dict'])
        self.critic_target.load_state_dict(ckpt['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)


__all__ = ['ContinuousSACAgent', 'ContinuousPPOAgent', 'TD3Agent']
