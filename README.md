# Go1 Handstand Policy Visualization

A streamlined toolkit for visualizing trained Go1 handstand policies in MuJoCo. This system enables real-time interactive viewing and video rendering of reinforcement learning policies trained using Brax PPO.

## Overview

This package provides tools to:
- Visualize trained policies in an interactive MuJoCo viewer
- Render policy execution to high-quality video files
- Analyze policy performance across multiple episodes
- Compare different training checkpoints

## System Requirements

- **Python Environment**: `~/mujoco_env` with MuJoCo, Brax, and JAX
- **Graphics Backend**: GLFW for interactive rendering
- **Dependencies**: `unitree_rl_mugym` repository

## Directory Structure

```
go1_handstand_render/
├── README.md                    # Documentation
├── run.sh                       # Main launcher
├── viewer.py                    # Interactive visualization
├── render.py                    # Video rendering
├── checkpoints/                 # Trained policy checkpoints
│   └── Go1Handstand_checkpoint/
│       ├── 0/
│       ├── 10000/
│       ├── 25067520/
│       ├── 50135040/
│       ├── 75202560/
│       └── 100270080/           # Latest checkpoint
└── videos/                      # Generated videos (auto-created)
```

## Quick Start

### Step 1: Download Checkpoint from Colab

```python
# In Colab notebook after training
from google.colab import files
import shutil

# Create checkpoint archive
shutil.make_archive('Go1Handstand_checkpoint', 'zip',
                    'path/to/checkpoint/directory')
files.download('Go1Handstand_checkpoint.zip')
```

### Step 2: Extract Locally

```bash
cd ~/go1_handstand_render
unzip ~/Downloads/Go1Handstand_checkpoint.zip -d checkpoints/
```

### Step 3: Launch Viewer

```bash
cd ~/go1_handstand_render
./run.sh
```

## Usage

### Interactive Viewer

Real-time visualization with camera control.

```bash
# Launch via menu
./run.sh
# Select option 1

# Direct execution
~/mujoco_env/bin/python3 viewer.py

# Specific checkpoint
~/mujoco_env/bin/python3 viewer.py checkpoints/Go1Handstand_checkpoint/50135040
```

**Controls:**
- Left Mouse: Rotate camera
- Right Mouse: Pan camera
- Scroll Wheel: Zoom
- Double-Click: Track body
- ESC: Exit

**Features:**
- Continuous episode playback with automatic reset
- Real-time performance metrics (steps, reward, duration)
- 60 FPS rendering

### Video Rendering

Generate MP4 files for documentation or analysis.

```bash
# Render 3 episodes (default)
~/mujoco_env/bin/python3 render.py

# Render 10 episodes
~/mujoco_env/bin/python3 render.py 10

# Specific checkpoint with 5 episodes
~/mujoco_env/bin/python3 render.py checkpoints/Go1Handstand_checkpoint/25067520 5
```

**Output Specifications:**
- Resolution: 1280x720 (720p)
- Frame Rate: 50 FPS
- Format: H.264 MP4
- Location: `videos/`
- Naming: `handstand_step{checkpoint}_ep{episode}.mp4`

## Checkpoint Management

### Automatic Selection

Scripts automatically use the checkpoint with the highest training step count.

### Manual Selection

```bash
# Compare training progression
~/mujoco_env/bin/python3 viewer.py checkpoints/Go1Handstand_checkpoint/10000    # Early
~/mujoco_env/bin/python3 viewer.py checkpoints/Go1Handstand_checkpoint/50135040 # Mid
~/mujoco_env/bin/python3 viewer.py checkpoints/Go1Handstand_checkpoint/100270080 # Final
```

### Multiple Checkpoints

Store multiple checkpoint directories for easy comparison:

```
checkpoints/
├── Go1Handstand_checkpoint/      # Latest training run
├── Go1Handstand_v2/              # Experimental config
└── Go1Handstand_baseline/        # Baseline comparison
```

## Expected Performance

Typical metrics for a well-trained policy (100M steps):

| Metric | Value |
|--------|-------|
| Average Episode Length | 40-60 steps |
| Best Episodes | 100-130 steps |
| Average Reward/Step | 0.014-0.018 |
| Success Rate | Variable (handstand is unstable) |

Episodes terminate when the robot falls or after 500 steps. Handstand tasks are inherently unstable, so early termination is expected behavior.

## Technical Details

### Network Architecture

- Policy Network: 512 → 256 → 128 (fully connected)
- Value Network: 512 → 256 → 128 (fully connected)
- Algorithm: Proximal Policy Optimization (PPO)
- Framework: Brax with MuJoCo MJX backend

### Environment Specifications

- Task: Go1Handstand
- Robot: Unitree Go1 quadruped
- Observation Space: 45D state + 94D privileged info
- Action Space: 12D (joint torques)
- Control Frequency: 50 Hz

### Rendering

- Interactive Mode: GLFW via MuJoCo passive viewer
- Video Mode: MuJoCo offscreen renderer with mediapy
- Physics: MuJoCo forward kinematics (visualization only)

## Troubleshooting

### Viewer Closes Immediately

Ensure GLFW backend is set:
```bash
export MUJOCO_GL=glfw
~/mujoco_env/bin/python3 viewer.py
```

### Import Errors

Verify unitree_rl_mugym is accessible:
```bash
ls ~/unitree_rl_mugym/mujoco_playground
```

### Slow Initial Rendering

First episode experiences JIT compilation overhead (approximately 30 seconds). Subsequent episodes run at full speed.

### Missing Checkpoints

Verify checkpoint structure:
```bash
ls ~/go1_handstand_render/checkpoints/Go1Handstand_checkpoint/
# Expected: 0/ 10000/ 25067520/ 50135040/ 75202560/ 100270080/ config.json
```

## Workflow Integration

### Training Pipeline

1. Train policy on Google Colab
2. Download checkpoint at desired intervals
3. Extract to checkpoints directory
4. Visualize or render locally
5. Analyze performance metrics
6. Iterate on training configuration

### Documentation

Use video rendering for:
- Research presentations
- Technical documentation
- Performance demonstrations
- Debugging analysis

## References

- Environment: mujoco_playground/unitree_rl_mugym
- Training Algorithm: Brax PPO
- Simulation: MuJoCo 3.x
- Robot Model: Unitree Go1

---

**Version**: 1.0
**Last Updated**: 2025-11-02
