#!/usr/bin/env python3
"""
Evaluate all checkpoints to find the best performing one
"""

import jax
import mujoco
import sys
import os
from pathlib import Path
import numpy as np

def evaluate_checkpoint(checkpoint_path, num_episodes=10):
    """Evaluate a single checkpoint across multiple episodes"""

    # Import modules
    sys.path.insert(0, str(Path.home() / "unitree_rl_mugym"))
    from mujoco_playground import registry, wrapper
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    from functools import partial

    # Load environment
    env_name = 'PandaPickCubeOrientation'
    env_cfg = registry.get_default_config(env_name)
    eval_env = registry.load(env_name, config=env_cfg)

    # Load checkpoint
    network_factory = partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(32, 32, 32, 32),
        value_hidden_layer_sizes=(256, 256, 256, 256, 256)
    )

    make_inference_fn, params, metrics = ppo.train(
        environment=registry.load(env_name, config=env_cfg),
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=str(checkpoint_path),
        num_timesteps=0,
        num_evals=0,
        episode_length=500,
        network_factory=network_factory,
        seed=42,
    )

    # Setup inference
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    # Run episodes
    rng = jax.random.PRNGKey(42)
    episode_lengths = []
    episode_rewards = []

    for ep in range(num_episodes):
        state = jit_reset(rng)
        total_reward = 0.0
        steps = 0

        while not state.done and steps < 500:
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            total_reward += float(state.reward)
            steps += 1

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        print(f"  Episode {ep+1:2d}: {steps:3d} steps, reward: {total_reward:7.3f}")

    return {
        'episode_lengths': episode_lengths,
        'episode_rewards': episode_rewards,
        'avg_length': np.mean(episode_lengths),
        'avg_reward': np.mean(episode_rewards),
        'std_length': np.std(episode_lengths),
        'std_reward': np.std(episode_rewards),
    }

def main():
    checkpoint_dir = Path(__file__).parent / "checkpoints" / "frankapickcube_checkpoint"

    checkpoints = [
        checkpoint_dir / "000006717440",
        checkpoint_dir / "000013434880",
        checkpoint_dir / "000020152320",
    ]

    print("="*70)
    print(" "*20 + "CHECKPOINT EVALUATION")
    print("="*70)
    print(f"Testing {len(checkpoints)} checkpoints with 10 episodes each")
    print("="*70 + "\n")

    results = {}

    for cp in checkpoints:
        if not cp.exists():
            print(f"Checkpoint not found: {cp}")
            continue

        print(f"Evaluating: {cp.name}")
        print("-" * 70)

        metrics = evaluate_checkpoint(cp, num_episodes=10)
        results[cp.name] = metrics

        print(f"\nResults:")
        print(f"  Avg Episode Length: {metrics['avg_length']:.1f} ± {metrics['std_length']:.1f}")
        print(f"  Avg Total Reward:   {metrics['avg_reward']:.3f} ± {metrics['std_reward']:.3f}")
        print(f"  Avg Reward/Step:    {metrics['avg_reward']/metrics['avg_length']:.4f}")
        print("\n")

    # Summary comparison
    print("="*70)
    print(" "*25 + "COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Checkpoint':<15} {'Avg Length':<15} {'Avg Reward':<15} {'Reward/Step':<15}")
    print("-" * 70)

    best_length = None
    best_reward = None
    best_checkpoint_length = None
    best_checkpoint_reward = None

    for name, metrics in sorted(results.items()):
        reward_per_step = metrics['avg_reward'] / metrics['avg_length']
        print(f"{name:<15} {metrics['avg_length']:<15.1f} {metrics['avg_reward']:<15.3f} {reward_per_step:<15.4f}")

        if best_length is None or metrics['avg_length'] > best_length:
            best_length = metrics['avg_length']
            best_checkpoint_length = name

        if best_reward is None or metrics['avg_reward'] > best_reward:
            best_reward = metrics['avg_reward']
            best_checkpoint_reward = name

    print("="*70)
    print("\nBEST PERFORMERS:")
    print(f"  Longest Episodes:  {best_checkpoint_length} ({best_length:.1f} steps avg)")
    print(f"  Highest Reward:    {best_checkpoint_reward} ({best_reward:.3f} avg)")
    print("="*70)

if __name__ == "__main__":
    main()
