#!/bin/bash
# Go1 Handstand Policy Viewer

PYTHON="$HOME/mujoco_env/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "     PandaPickCubeOrientation Policy Viewer"
echo "=========================================="
echo ""

# Check Python environment
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python environment not found at $PYTHON"
    echo "Please ensure mujoco_env is correctly installed."
    exit 1
fi

# Check checkpoints
if [ -z "$(ls -A "$SCRIPT_DIR/checkpoints" 2>/dev/null)" ]; then
    echo "ERROR: No checkpoints found in checkpoints/ directory"
    echo ""
    echo "Setup Instructions:"
    echo "  1. Download trained checkpoint from Colab"
    echo "  2. Extract to: $SCRIPT_DIR/checkpoints/"
    echo "  3. Run this script again"
    echo ""
    exit 1
fi

echo "✓ Environment verified"
echo "✓ Checkpoint detected"
echo ""
echo "Select mode:"
echo "  1) Interactive viewer (real-time visualization)"
echo "  2) Render to video (save MP4 files)"
echo ""
read -p "Choice [1]: " choice
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "Launching interactive viewer..."
        "$PYTHON" "$SCRIPT_DIR/viewer.py"
        ;;
    2)
        echo ""
        read -p "Number of episodes to render [3]: " num_eps
        num_eps=${num_eps:-3}
        echo ""
        echo "Rendering $num_eps episodes..."
        "$PYTHON" "$SCRIPT_DIR/render.py" $num_eps
        ;;
    *)
        echo "Invalid selection"
        exit 1
        ;;
esac
