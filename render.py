#!/usr/bin/env python3
"""
Go1 Handstand - Video Renderer
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
        checkpoints = sorted(checkpoint_dir.glob("*/"))
        if not checkpoints:
            print("ERROR: No checkpoints found in checkpoints/ folder!")
            sys.exit(1)
        checkpoint_path = max(checkpoints, key=lambda p: int(p.name) if p.name.isdigit() else 0)

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

    # Load environment
    print("Loading environment...")
    env_name = 'PandaPickCubeOrientation'
    env_cfg = registry.get_default_config(env_name)
    eval_env = registry.load(env_name, config=env_cfg)

    # Load checkpoint
    print("Loading checkpoint...")
    network_factory = partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128)
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

    # Get MuJoCo model
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Setup renderer (720p)
    renderer = mujoco.Renderer(mj_model, height=720, width=1280)

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

        while not state.done and step < 500:
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
                print(f"  Rendering... {step}/500 steps")

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
