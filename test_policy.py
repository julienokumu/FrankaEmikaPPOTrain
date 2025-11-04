#!/usr/bin/env python3
"""Test script to verify policy performance with normalization"""

import jax
import sys
from pathlib import Path
import numpy as np

# Setup paths
sys.path.insert(0, str(Path.home() / "unitree_rl_mugym"))
from mujoco_playground import registry, wrapper
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from functools import partial

# Find checkpoint
checkpoint_dir = Path(__file__).parent / "checkpoints"
all_checkpoints = []
for checkpoint_root in checkpoint_dir.glob("*/"):
    numbered_subdirs = [d for d in checkpoint_root.glob("*/") if d.name.replace('_', '').isdigit()]
    if numbered_subdirs:
        all_checkpoints.extend(numbered_subdirs)

checkpoint_path = max(all_checkpoints, key=lambda p: int(p.name.replace('_', '')))

print("="*70)
print("POLICY PERFORMANCE TEST")
print("="*70)
print(f"Checkpoint: {checkpoint_path.name}\n")

# Load environment with custom config
env_name = 'PandaPickCubeOrientation'
env_cfg = registry.get_default_config(env_name)
env_cfg['action_scale'] = 0.04
env_cfg['ctrl_dt'] = 0.02
env_cfg['sim_dt'] = 0.005
env_cfg['reward_config'] = {
    'scales': {
        'box_target': 8.0,
        'gripper_box': 4.0,
        'no_floor_collision': 0.25,
        'robot_target_qpos': 0.3
    }
}

eval_env = registry.load(env_name, config=env_cfg)

# Load checkpoint
network_factory = partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(32, 32, 32, 32),
    value_hidden_layer_sizes=(256, 256, 256, 256, 256)
)

print("Loading checkpoint...")
make_inference_fn, params, _ = ppo.train(
    environment=registry.load(env_name, config=env_cfg),
    eval_env=eval_env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    num_envs=1,
    normalize_observations=True,
    restore_checkpoint_path=str(checkpoint_path),
    num_timesteps=0,
    num_evals=0,
    episode_length=150,
    network_factory=network_factory,
    seed=42,
)

# Extract normalization parameters
if '0' in params and 'mean' in params['0'] and 'std' in params['0']:
    norm_mean = np.array(params['0']['mean'])
    norm_std = np.array(params['0']['std'])
    print(f"✓ Loaded normalization stats")
    print(f"  Mean range: [{norm_mean.min():.2f}, {norm_mean.max():.2f}]")
    print(f"  Std range: [{norm_std.min():.2f}, {norm_std.max():.2f}]")
    use_norm = True
else:
    print("✗ No normalization parameters found!")
    norm_mean = None
    norm_std = None
    use_norm = False

# Setup inference
inference_fn = make_inference_fn(params, deterministic=True)

if use_norm:
    def normalized_inference_fn(obs, rng):
        normalized_obs = (obs - norm_mean) / (norm_std + 1e-8)
        return inference_fn(normalized_obs, rng)
    jit_inference_fn = jax.jit(normalized_inference_fn)
    print("✓ Using manual normalization\n")
else:
    jit_inference_fn = jax.jit(inference_fn)
    print("✗ Using raw observations\n")

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# Test 5 episodes
print("="*70)
print("Running 5 test episodes...")
print("="*70)

rng = jax.random.PRNGKey(42)
rewards = []

for ep in range(5):
    state = jit_reset(rng)
    total_reward = 0.0
    step = 0

    while not state.done and step < 150:
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        total_reward += float(state.reward)
        step += 1

    rewards.append(total_reward)
    print(f"Episode {ep+1}: Steps={step:4d}, Total Reward={total_reward:8.2f}, Avg={total_reward/step:.4f}")

print("\n" + "="*70)
print(f"Average total reward: {np.mean(rewards):.2f}")
print(f"Std deviation: {np.std(rewards):.2f}")
print("="*70)

# For 150-step episodes, expect roughly 1/7th of the 1000-step reward
if np.mean(rewards) > 800:
    print("✓ SUCCESS: Policy is performing well!")
elif np.mean(rewards) > 200:
    print("✗ PARTIAL: Policy is underperforming (normalization might not be working)")
else:
    print("✗ FAILURE: Policy is performing very poorly")
