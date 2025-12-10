"""Train continuous-control RL agents on Gymnasium environments with WandB logging."""

import argparse
import os
from typing import Dict, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
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


def make_env(env_name, seed=None, record_video=False, video_folder="videos", algo=None, video_frequency=50):
    render_mode = 'rgb_array' if record_video else None
    env_kwargs = CONTINUOUS_ENV_KWARGS.get(env_name, {}).copy()
    env = gym.make(env_name, render_mode=render_mode,**env_kwargs) #type: ignore
    if seed is not None:
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass
    if record_video:
        video_path = os.path.join(video_folder, algo.upper() if algo else "", env_name)
        os.makedirs(video_path, exist_ok=True)
        def should_record(episode_id):
            return episode_id == 0 or ((episode_id + 1) % video_frequency == 0)
        env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=should_record, name_prefix=env_name)
    return env


def train(
    env_name: str,
    algo: str,
    episodes: int = 500,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    record_video: bool = False,
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
    load_model: Optional[str] = None,
    start_episode: int = 0,
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
    }

    run = wandb.init(project=project, entity=entity, name=run_name, config=config)

    env = make_env(env_name, seed=seed, record_video=record_video, algo=algo)
    action_space = env.action_space
    if not isinstance(action_space, spaces.Box):
        raise ValueError("Continuous training script expects a continuous (Box) action space")

    obs, _ = env.reset(seed=seed)
    obs_vector = flatten_obs(obs)
    n_observations = obs_vector.shape[0]

    algo_lower = algo.lower()
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

    if load_model:
        print(f"Loading model from {load_model}...")
        agent.load(load_model)

    print(f"Training {algo.upper()} on {env_name} for {episodes} episodes...")
    print(f"Device: {torch_device.type}, Action dim: {action_space.shape[0]}, Observations: {n_observations}")

    best_reward = -float('inf')
    for i_episode in range(start_episode, episodes):
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
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--learning-rate', '--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=0.0)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--record-video', action='store_true')
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
    parser.add_argument('--save-best', action='store_true')
    parser.add_argument('--load-model', type=str, default=None, help='Path to model checkpoint to load')
    parser.add_argument('--start-episode', type=int, default=0, help='Episode number to start from')
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
        load_model=args.load_model,
        start_episode=args.start_episode,
    )
