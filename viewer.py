#!/usr/bin/env python3
"""
Panda Pick Cube Orientation - Interactive MuJoCo Viewer
View your trained policy in real-time
"""

import jax
import mujoco
import mujoco.viewer
import sys
import os
import time
from pathlib import Path
import numpy as np

# IMPORTANT: Must set before importing MuJoCo
os.environ['MUJOCO_GL'] = 'glfw'

def main():
    # Get checkpoint path from command line or use default
    if len(sys.argv) > 1:
        checkpoint_path = Path(sys.argv[1])
    else:
        # Look for latest checkpoint in checkpoints folder
        checkpoint_dir = Path(__file__).parent / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("*/"))
        if not checkpoints:
            print("ERROR: No checkpoints found in checkpoints/ folder!")
            print("\nPlease:")
            print("  1. Download your trained checkpoint from Colab")
            print("  2. Extract it to: ~/go1_handstand_render/checkpoints/")
            print("  3. Run this script again")
            sys.exit(1)

        # Use the checkpoint with the highest step number
        checkpoint_path = max(checkpoints, key=lambda p: int(p.name) if p.name.isdigit() else 0)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"Using checkpoint: {checkpoint_path.name}")

    # Import modules
    sys.path.insert(0, str(Path.home() / "unitree_rl_mugym"))
    from mujoco_playground import registry, wrapper
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    from functools import partial

    print("\n" + "="*70)
    print(" "*15 + "PandaPickCubeOrientation - Interactive Viewer")
    print("="*70)

    # Load environment
    print("\n[1/4] Loading environment...")
    env_name = 'PandaPickCubeOrientation'
    env_cfg = registry.get_default_config(env_name)
    eval_env = registry.load(env_name, config=env_cfg)
    print("      ✓ Environment loaded")

    # Load checkpoint
    print("[2/4] Loading trained policy checkpoint...")
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
    print(f"      ✓ Policy loaded from step {checkpoint_path.name}")

    # Setup inference
    print("[3/4] Setting up real-time inference...")
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(42)
    brax_state = jit_reset(rng)

    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    print("      ✓ Inference ready")

    print("[4/4] Launching MuJoCo viewer...")
    print("\n" + "="*70)
    print("                         VIEWER CONTROLS")
    print("="*70)
    print("  Left Mouse Button   : Rotate camera")
    print("  Right Mouse Button  : Pan camera")
    print("  Scroll Wheel       : Zoom in/out")
    print("  Double-click       : Select/track body")
    print("  ESC / Close Window : Exit")
    print("="*70)
    print("\nThe viewer will run continuously until you close it.")
    print("Episodes will automatically reset when the robot falls.\n")
    print("="*70 + "\n")

    step = 0
    episode = 1
    episode_reward = 0.0
    episode_start_time = time.time()

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        print("✓ Viewer window opened - Watch your robot perform handstands!\n")

        # Continuous loop - runs until you close the window
        while True:
            loop_start = time.time()

            # Get action from policy
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(brax_state.obs, act_rng)

            # Step environment
            brax_state = jit_step(brax_state, ctrl)
            episode_reward += float(brax_state.reward)

            # Update MuJoCo visualization
            mj_data.qpos[:] = np.array(brax_state.data.qpos)
            mj_data.qvel[:] = np.array(brax_state.data.qvel)
            mj_data.ctrl[:] = np.array(ctrl)
            mujoco.mj_forward(mj_model, mj_data)

            # Sync viewer
            try:
                viewer.sync()
            except:
                # Viewer was closed
                break

            step += 1

            # Episode complete - reset
            if brax_state.done or step >= 500:
                episode_duration = time.time() - episode_start_time
                avg_reward = episode_reward / step

                print(f"Episode {episode:3d} | "
                      f"Steps: {step:3d} | "
                      f"Reward: {episode_reward:6.3f} | "
                      f"Avg: {avg_reward:.4f} | "
                      f"Duration: {episode_duration:.1f}s")

                # Reset for next episode
                brax_state = jit_reset(rng)
                step = 0
                episode += 1
                episode_reward = 0.0
                episode_start_time = time.time()

            # Frame rate control (~60 FPS)
            elapsed = time.time() - loop_start
            sleep_time = max(0, 0.016 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\n" + "="*70)
    print(f"Viewer closed after {episode-1} episodes")
    print("="*70)

if __name__ == "__main__":
    main()
