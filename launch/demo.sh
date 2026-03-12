#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="${PIXI_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

# Default parameters (1080p for testing)
WIDTH=1920
HEIGHT=1080
BACKEND="cuda"
PUBLISH_RATE=4
RECORD_PATH=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --rate)
            PUBLISH_RATE="$2"
            shift 2
            ;;
        --record)
            RECORD_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --width WIDTH         Image width (default: 1920 for 1080p)"
            echo "  --height HEIGHT       Image height (default: 1080)"
            echo "  --backend BACKEND     cuda or cpu (default: cuda)"
            echo "  --rate RATE_MS        Publish rate in ms (default: 1)"
            echo "  --record PATH         Record video to MP4 file (requires ffmpeg)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate backend
if [[ "$BACKEND" != "cuda" && "$BACKEND" != "cpu" ]]; then
    echo "Error: backend must be 'cuda' or 'cpu', got: $BACKEND"
    exit 1
fi

USE_CUDA="true"
if [[ "$BACKEND" == "cpu" ]]; then
    USE_CUDA="false"
fi

. "$WS_ROOT/build/torch_backend_demo/colcon_command_prefix_build.sh"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export LD_LIBRARY_PATH="$WS_ROOT/build/torch_backend_demo:$WS_ROOT/install/torch_backend_demo/lib:$WS_ROOT/.pixi/envs/default/libtorch/lib:${LD_LIBRARY_PATH:-}"

ZENOHD="$WS_ROOT/install/rmw_zenoh_cpp/lib/rmw_zenoh_cpp/rmw_zenohd"
RENDERER="$WS_ROOT/build/torch_backend_demo/renderer_node"
DISPLAY_NODE="$WS_ROOT/build/torch_backend_demo/display_node"

cleanup() {
    kill -INT $DISPLAY_PID $RENDERER_PID 2>/dev/null
    wait $DISPLAY_PID $RENDERER_PID 2>/dev/null
    if [[ -n "${ZENOH_PID:-}" ]]; then
        kill $ZENOH_PID 2>/dev/null
        wait $ZENOH_PID 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

ZENOH_PID=""
if ss -tlnH 2>/dev/null | grep -q ':7447 '; then
    echo "Zenoh router already on 7447, reusing it."
else
    echo "Starting zenoh router..."
    $ZENOHD &
    ZENOH_PID=$!
    sleep 1
fi

echo "Starting renderer (publisher)..."
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Backend: $BACKEND"
echo "  Rate: ${PUBLISH_RATE}ms"
$RENDERER --ros-args \
    -p image_width:=$WIDTH \
    -p image_height:=$HEIGHT \
    -p use_cuda:=$USE_CUDA \
    -p publish_rate_ms:=$PUBLISH_RATE &
RENDERER_PID=$!

DISPLAY_ARGS=""
if [[ -n "$RECORD_PATH" ]]; then
    DISPLAY_ARGS="--ros-args -p record_path:=$RECORD_PATH"
    echo "Starting display (subscriber, recording to $RECORD_PATH)..."
else
    echo "Starting display (subscriber)..."
fi
$DISPLAY_NODE $DISPLAY_ARGS &
DISPLAY_PID=$!

echo "All nodes running. Press Ctrl+C to stop."
wait
