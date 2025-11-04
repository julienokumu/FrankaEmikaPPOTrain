# Franka Emika Panda PPO Training & Visualization

A complete toolkit for training and visualizing Franka Emika Panda robot arm pick-and-place policies using Proximal Policy Optimization (PPO) in MuJoCo. This repository includes the training notebook, visualization tools, and pre-configured scripts for both interactive viewing and video rendering.

## Overview

This project provides end-to-end tooling for:
- **Training**: PPO-based reinforcement learning for robotic manipulation (Colab notebook included)
- **Visualization**: Real-time interactive viewing of trained policies in MuJoCo
- **Video Rendering**: High-quality MP4 generation for documentation and analysis
- **Evaluation**: Performance testing and checkpoint comparison tools

## Table of Contents

- [Training Configuration](#training-configuration)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Interactive Viewer](#interactive-viewer)
  - [Video Rendering](#video-rendering)
  - [Policy Testing](#policy-testing)
- [Repository Structure](#repository-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

## Training Configuration

This policy was trained using the following hyperparameters:

### PPO Hyperparameters

```yaml
action_repeat: 1
batch_size: 1024
discounting: 0.97
entropy_cost: 0.02
episode_length: 150
learning_rate: 0.005
num_envs: 2048
num_evals: 4
num_minibatches: 32
num_timesteps: 20000000
num_updates_per_batch: 8
reward_scaling: 1.0
unroll_length: 10
normalize_observations: true
```

### Network Architecture

```yaml
network_factory:
  policy_hidden_layer_sizes: [32, 32, 32, 32]
  value_hidden_layer_sizes: [256, 256, 256, 256, 256]
  policy_obs_key: state
  value_obs_key: state
```

### Environment Configuration

```yaml
action_scale: 0.04
ctrl_dt: 0.02
sim_dt: 0.005
episode_length: 150

reward_config:
  scales:
    box_target: 8.0
    gripper_box: 4.0
    no_floor_collision: 0.25
    robot_target_qpos: 0.3
```

**Training Notebook**: The complete training pipeline is included in `FrankaPandaPPOTrain.ipynb`, which can be run on Google Colab with GPU/TPU acceleration.

## Quick Start

### Prerequisites

- Python 3.8+
- MuJoCo 3.x
- CUDA-compatible GPU (for training)
- Linux/macOS (recommended for visualization)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/FrankaEmikaPPOTrain.git
cd FrankaEmikaPPOTrain
```

2. **Install dependencies**:
```bash
# Create virtual environment
python3 -m venv mujoco_env
source mujoco_env/bin/activate  # On Windows: mujoco_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

3. **Install unitree_rl_mugym**:
```bash
cd ~
git clone https://github.com/unitree/unitree_rl_mugym.git
cd unitree_rl_mugym
pip install -e .
```

4. **Download trained checkpoint** from Colab or use your own:
```bash
# Extract checkpoint to checkpoints/ directory
unzip your_checkpoint.zip -d checkpoints/
```

5. **Launch viewer**:
```bash
./run.sh
```

## Installation

### Full Installation Steps

```bash
# 1. System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip git libglfw3 libglfw3-dev

# 2. Create Python environment
python3 -m venv ~/mujoco_env
source ~/mujoco_env/bin/activate

# 3. Install Python packages
pip install --upgrade pip
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install mujoco brax dm-control mediapy numpy

# 4. Clone and install unitree_rl_mugym
cd ~
git clone https://github.com/unitree/unitree_rl_mugym.git
cd unitree_rl_mugym
pip install -e .

# 5. Clone this repository
cd ~
git clone https://github.com/yourusername/FrankaEmikaPPOTrain.git
cd FrankaEmikaPPOTrain
```

## Usage

### Interactive Viewer

Visualize the trained policy in real-time with interactive camera controls.

```bash
# Launch via menu system
./run.sh

# Direct execution (auto-selects latest checkpoint)
~/mujoco_env/bin/python3 viewer.py

# Specify checkpoint manually
~/mujoco_env/bin/python3 viewer.py checkpoints/frankapickcube_checkpoint/000020000000
```

**Viewer Controls**:
- **Left Mouse Button**: Rotate camera
- **Right Mouse Button**: Pan camera
- **Scroll Wheel**: Zoom in/out
- **Double-click**: Select/track body
- **ESC**: Exit viewer

**Features**:
- Continuous episode playback with automatic reset
- Real-time performance metrics (steps, reward, average reward)
- 60 FPS smooth rendering
- Episodes run for 150 steps or until task completion

### Video Rendering

Generate MP4 video files for documentation, presentations, or analysis.

```bash
# Render 3 episodes (default)
~/mujoco_env/bin/python3 render.py

# Render custom number of episodes
~/mujoco_env/bin/python3 render.py 10

# Render from specific checkpoint
~/mujoco_env/bin/python3 render.py checkpoints/frankapickcube_checkpoint/000020000000 5
```

**Video Specifications**:
- **Resolution**: 640x480 (480p)
- **Frame Rate**: 50 FPS
- **Format**: H.264 MP4
- **Output Directory**: `videos/`
- **Naming Convention**: `pickcube_step{checkpoint}_ep{episode}.mp4`

### Policy Testing

Test policy performance across multiple episodes without visualization.

```bash
~/mujoco_env/bin/python3 test_policy.py
```

This script:
- Loads the latest checkpoint
- Runs 5 test episodes
- Reports average reward, standard deviation
- Validates policy performance
- Checks if normalization is working correctly

## Repository Structure

```
FrankaEmikaPPOTrain/
├── README.md                      # This file
├── FrankaPandaPPOTrain.ipynb     # Training notebook (Google Colab)
├── requirements.txt               # Python dependencies
├── run.sh                         # Interactive launcher script
│
├── viewer.py                      # Interactive MuJoCo viewer
├── render.py                      # Video rendering script
├── test_policy.py                 # Policy evaluation tool
├── evaluate_checkpoints.py        # Multi-checkpoint comparison
│
├── checkpoints/                   # Trained model checkpoints
│   └── frankapickcube_checkpoint/
│       ├── config.json
│       ├── 000006717440/         # 6.7M steps
│       ├── 000013434880/         # 13.4M steps
│       └── 000020000000/         # 20M steps (final)
│
└── videos/                        # Generated video files (auto-created)
    └── *.mp4
```

## Technical Details

### Environment

- **Task**: PandaPickCubeOrientation
- **Robot**: Franka Emika Panda (7-DOF arm + parallel gripper)
- **Observation Space**: 66-dimensional state vector
- **Action Space**: 8-dimensional (7 joint velocities + gripper)
- **Control Frequency**: 50 Hz (ctrl_dt: 0.02s)
- **Physics Timestep**: 200 Hz (sim_dt: 0.005s)

### Algorithm

- **Method**: Proximal Policy Optimization (PPO)
- **Framework**: Brax with MuJoCo MJX backend
- **Training Time**: ~20M environment steps
- **Parallel Environments**: 2048
- **Batch Size**: 1024
- **Observation Normalization**: Enabled (running mean/std)

### Network Architecture

**Policy Network**:
- 4 hidden layers of 32 units each
- Activation: tanh (implicit in PPO implementation)
- Output: Continuous action distribution (tanh normal)

**Value Network**:
- 5 hidden layers of 256 units each
- Output: Single value estimate

### Rendering

- **Interactive Mode**: GLFW via MuJoCo passive viewer
- **Batch Mode**: MuJoCo offscreen renderer
- **Video Encoding**: H.264 via mediapy/ffmpeg
- **Physics**: MuJoCo forward kinematics for visualization

## Troubleshooting

### Viewer Closes Immediately

Ensure GLFW graphics backend is set:
```bash
export MUJOCO_GL=glfw
~/mujoco_env/bin/python3 viewer.py
```

### Import Errors

Verify `unitree_rl_mugym` is installed:
```bash
python3 -c "from mujoco_playground import registry; print('OK')"
```

If this fails:
```bash
cd ~/unitree_rl_mugym
pip install -e .
```

### Low Policy Performance

If the policy performs poorly (low rewards, doesn't pick cube):

1. **Check environment config matches training**:
   - Episode length should be 150
   - Action scale should be 0.04
   - Environment config must match training parameters

2. **Verify checkpoint loaded correctly**:
   - Look for "✓ Policy loaded from step..." message
   - Check that normalization stats are loaded

3. **Test with test_policy.py**:
```bash
~/mujoco_env/bin/python3 test_policy.py
```
Should report average reward > 1000 for successful policy.

### CUDA Out of Memory (Training)

In the Colab notebook, reduce:
- `num_envs`: 2048 → 1024
- `batch_size`: 1024 → 512

### Overflow Warnings

The warning `/site-packages/jax/_src/abstract_arrays.py:135: RuntimeWarning: overflow encountered in cast` is harmless and doesn't affect policy performance. It occurs during checkpoint loading and can be ignored.

### Missing Dependencies

If you encounter missing module errors:
```bash
pip install jax mujoco brax mediapy numpy dm-control
```

For GPU support (training):
```bash
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Training from Scratch

To train your own policy:

1. Open `FrankaPandaPPOTrain.ipynb` in Google Colab
2. Enable GPU/TPU runtime (Runtime → Change runtime type)
3. Run all cells sequentially
4. Adjust hyperparameters in the configuration cell if desired
5. Training takes approximately 2-4 hours for 20M steps on Colab TPU
6. Download checkpoint when training completes
7. Extract to `checkpoints/` directory in this repository

## Expected Performance

For a well-trained policy (20M steps):

| Metric | Value |
|--------|-------|
| Episode Length | 150 steps |
| Average Total Reward | 1000-1500 |
| Average Reward per Step | 7-10 |
| Success Rate | >80% cube pickups |
| Training Time | 2-4 hours (Colab TPU) |

## Citation

If you use this code in your research, please cite:

```bibtex
@software{franka_ppo_train_2025,
  author = {Your Name},
  title = {Franka Emika Panda PPO Training & Visualization},
  year = {2025},
  url = {https://github.com/yourusername/FrankaEmikaPPOTrain}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Brax**: Google's differentiable physics engine for reinforcement learning
- **MuJoCo**: Advanced physics simulation by DeepMind/Roboti
- **unitree_rl_mugym**: Gymnasium-style environments for Unitree robots and manipulation tasks
- **Franka Emika**: Panda robot arm platform

---

**Version**: 2.0
**Last Updated**: 2025-11-04
**Maintained by**: [Your Name]
