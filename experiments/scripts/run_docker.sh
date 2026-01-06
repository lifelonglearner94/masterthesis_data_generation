#!/bin/bash
# =============================================================================
# run_docker.sh - Launch Kubric Docker container with GPU support
# =============================================================================
#
# This script starts the official Kubric Docker container with:
# - NVIDIA GPU passthrough (--gpus all) when available
# - Workspace mounted at /workspace
# - Output directory mounted for persistent storage
#
# Usage:
#   ./run_docker.sh                    # Interactive shell
#   ./run_docker.sh --no-gpu           # Force CPU-only mode
#   ./run_docker.sh python script.py   # Run a specific command
#
# =============================================================================

set -e

# Configuration
KUBRIC_IMAGE="kubricdockerhub/kubruntu:latest"
# Go up two levels: scripts -> experiments -> project root
WORKSPACE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${WORKSPACE_DIR}/experiments/output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
USE_GPU=true
REMAINING_ARGS=()

for arg in "$@"; do
    case $arg in
        --no-gpu|--cpu)
            USE_GPU=false
            ;;
        *)
            REMAINING_ARGS+=("$arg")
            ;;
    esac
done

echo -e "${GREEN}Kubric Docker Launcher${NC}"
echo "========================"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check if running in WSL
IS_WSL=false
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=true
    echo -e "${YELLOW}WSL environment detected${NC}"
fi

# Check for GPU availability
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -q .; then
        GPU_AVAILABLE=true
        echo -e "${GREEN}GPU detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    fi
fi

# Decide whether to use GPU
if [ "$USE_GPU" = true ] && [ "$GPU_AVAILABLE" = false ]; then
    echo -e "${YELLOW}Warning: GPU not available. Running in CPU-only mode.${NC}"
    USE_GPU=false
fi

if [ "$USE_GPU" = false ]; then
    echo -e "${YELLOW}Running without GPU acceleration (CPU-only mode)${NC}"
fi

echo ""
echo "Workspace: ${WORKSPACE_DIR}"
echo "Output:    ${OUTPUT_DIR}"
echo ""

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Build Docker command
DOCKER_CMD=(
    docker run
    --rm
    -it
)

# Add GPU flag only if GPU is available and requested
if [ "$USE_GPU" = true ]; then
    DOCKER_CMD+=(--gpus all)
fi

DOCKER_CMD+=(
    -v "${WORKSPACE_DIR}:/workspace"
    -v "${OUTPUT_DIR}:/output"
    -w /workspace/experiments/scripts
    -e PYTHONPATH=/workspace
    "${KUBRIC_IMAGE}"
)

# Run with provided command or interactive shell
if [ ${#REMAINING_ARGS[@]} -eq 0 ]; then
    echo -e "${GREEN}Starting interactive shell...${NC}"
    "${DOCKER_CMD[@]}" /bin/bash
else
    echo -e "${GREEN}Running: ${REMAINING_ARGS[*]}${NC}"
    "${DOCKER_CMD[@]}" "${REMAINING_ARGS[@]}"
fi
