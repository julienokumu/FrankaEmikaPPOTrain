#!/usr/bin/env python3
"""
Panda Pick Cube Orientation - Video Renderer
Render episodes to MP4 files for sharing/analysis
"""

import jax
import mujoco
import sys
import os
from pathlib import Path
import numpy as np
import mediapy as media

def main():
    # Parse arguments
    num_episodes = 3
    if len(sys.argv) > 1:
        try:
            num_episodes = int(sys.argv[1])
        except ValueError:
            checkpoint_path = Path(sys.argv[1])
            if len(sys.argv) > 2:
                num_episodes = int(sys.argv[2])

    # Get checkpoint path
    if len(sys.argv) > 1 and not sys.argv[1].isdigit():
        checkpoint_path = Path(sys.argv[1])
    else:
        # Look for latest checkpoint
        checkpoint_dir = Path(__file__).parent / "checkpoints"

        # Find all numbered checkpoint directories (looking in subdirectories too)
        all_checkpoints = []
        for checkpoint_root in checkpoint_dir.glob("*/"):
            # Check if this directory contains numbered subdirectories (like 000020152320/)
            numbered_subdirs = [d for d in checkpoint_root.glob("*/") if d.name.replace('_', '').isdigit()]
            if numbered_subdirs:
                all_checkpoints.extend(numbered_subdirs)
            # Or check if this directory itself is a numbered checkpoint
            elif checkpoint_root.name.replace('_', '').isdigit():
                all_checkpoints.append(checkpoint_root)

        if not all_checkpoints:
            print("ERROR: No checkpoints found in checkpoints/ folder!")
            sys.exit(1)

        checkpoint_path = max(all_checkpoints, key=lambda p: int(p.name.replace('_', '')))

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    # Import modules
    sys.path.insert(0, str(Path.home() / "unitree_rl_mugym"))
    from mujoco_playground import registry, wrapper
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    from functools import partial

    print("="*70)
    print(" "*20 + "Panda Pick Cube Orientation Renderer")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Episodes to render: {num_episodes}")
    print("="*70 + "\n")

    # Load environment with custom config matching training
    print("Loading environment...")
    env_name = 'PandaPickCubeOrientation'
    env_cfg = registry.get_default_config(env_name)

    # Apply custom training config
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
    print("Loading checkpoint...")
    network_factory = partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(32, 32, 32, 32),
        value_hidden_layer_sizes=(256, 256, 256, 256, 256)
    )

    make_inference_fn, params, metrics = ppo.train(
        environment=registry.load(env_name, config=env_cfg),
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_envs=1,
        normalize_observations=True,
        restore_checkpoint_path=str(checkpoint_path),
        num_timesteps=0,
        num_evals=0,
        episode_length=150,  # Match training config
        network_factory=network_factory,
        seed=42,
    )

    # Extract normalization parameters
    if '0' in params and 'mean' in params['0'] and 'std' in params['0']:
        norm_mean = np.array(params['0']['mean'])
        norm_std = np.array(params['0']['std'])
        print(f"✓ Loaded normalization stats (mean range: [{norm_mean.min():.2f}, {norm_mean.max():.2f}])")
    else:
        print("WARNING: No normalization parameters found!")
        norm_mean = None
        norm_std = None

    # Setup inference
    inference_fn = make_inference_fn(params, deterministic=True)

    # Create normalized inference function
    if norm_mean is not None and norm_std is not None:
        def normalized_inference_fn(obs, rng):
            # Apply normalization: (obs - mean) / std
            normalized_obs = (obs - norm_mean) / (norm_std + 1e-8)
            return inference_fn(normalized_obs, rng)
        jit_inference_fn = jax.jit(normalized_inference_fn)
        print("✓ Using normalized observations")
    else:
        jit_inference_fn = jax.jit(inference_fn)
        print("WARNING: Using raw observations (no normalization)")

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    # Get MuJoCo model
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Setup renderer (480p - fits default framebuffer)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    print("✓ Setup complete\n")

    # Output directory
    output_dir = Path(__file__).parent / "videos"
    output_dir.mkdir(exist_ok=True)

    # Render episodes
    rng = jax.random.PRNGKey(42)

    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes}")
        print("-" * 70)

        state = jit_reset(rng)
        frames = []
        total_reward = 0.0
        step = 0

        while not state.done and step < 150:
            # Get action
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)

            # Step environment
            state = jit_step(state, ctrl)

            # Update MuJoCo visualization
            mj_data.qpos[:] = np.array(state.data.qpos)
            mj_data.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_forward(mj_model, mj_data)

            # Render frame
            renderer.update_scene(mj_data)
            frame = renderer.render()
            frames.append(frame)

            total_reward += float(state.reward)
            step += 1

            if step % 50 == 0:
                print(f"  Rendering... {step}/150 steps")

        # Save episode video
        video_name = f"pickcube_step{checkpoint_path.name}_ep{ep+1}.mp4"
        output_path = output_dir / video_name
        print(f"  Saving video: {video_name}")
        media.write_video(str(output_path), frames, fps=50)

        print(f"  ✓ Episode complete:")
        print(f"    Steps: {step}")
        print(f"    Total reward: {total_reward:.4f}")
        print(f"    Avg reward: {total_reward/step:.4f}")
        print(f"    Saved to: {output_path}")
        print()

    print("="*70)
    print("RENDERING COMPLETE!")
    print("="*70)
    print(f"\nGenerated {num_episodes} videos in: {output_dir}/")
    for ep in range(num_episodes):
        video_name = f"pickcube_step{checkpoint_path.name}_ep{ep+1}.mp4"
        print(f"  - {video_name}")
    print("="*70)

if __name__ == "__main__":
    main()
