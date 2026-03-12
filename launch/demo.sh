#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="${PIXI_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

resolve_resolution() {
    case "$1" in
        fhd|FHD|1080p) echo "1920 1080" ;;
        2k|2K|1440p)   echo "2560 1440" ;;
        4k|4K|2160p)   echo "3840 2160" ;;
        6k|6K)         echo "5760 3240" ;;
        *)
            echo "Error: unknown resolution '$1'. Use: fhd, 2k, 4k, 6k" >&2
            exit 1
            ;;
    esac
}

RESOLUTION="fhd"
BACKEND="cuda"
RECORD_PATH=""
WALLCLOCK="false"
COMPARE="false"
HEADLESS="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --record)
            RECORD_PATH="$2"
            shift 2
            ;;
        --wallclock)
            WALLCLOCK="true"
            shift
            ;;
        --compare)
            COMPARE="true"
            shift
            ;;
        --headless)
            HEADLESS="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --resolution RES      fhd (default), 2k, 4k, 6k"
            echo "  --backend BACKEND     cuda or cpu (default: cuda)"
            echo "  --record PATH         Record video to MP4 file (requires ffmpeg)"
            echo "  --wallclock           Use wall-clock timestep (default: fixed 1/60s)"
            echo "  --compare             Side-by-side CUDA vs CPU comparison"
            echo "  --headless            Run without display windows"
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

read -r WIDTH HEIGHT <<< "$(resolve_resolution "$RESOLUTION")"

if [[ "$COMPARE" == "true" ]]; then
    BACKEND="compare"
fi

if [[ "$BACKEND" != "cuda" && "$BACKEND" != "cpu" && "$BACKEND" != "compare" ]]; then
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

PIDS=()

cleanup() {
    for pid in "${PIDS[@]}"; do
        kill -INT "$pid" 2>/dev/null
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null
    done
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

TIMESTEP_STR="$( [[ "$WALLCLOCK" == "true" ]] && echo wallclock || echo fixed )"

if [[ "$COMPARE" == "true" ]]; then
    HALF_W=960
    HALF_H=540

    echo "=== Side-by-side comparison mode ==="
    echo "  Resolution: ${WIDTH}x${HEIGHT} ($RESOLUTION)"
    echo "  Window: ${HALF_W}x${HALF_H} each"
    echo "  Timestep: $TIMESTEP_STR"

    echo "Starting CUDA renderer + display (left window)..."
    $RENDERER --ros-args \
        -r __ns:=/cuda \
        -p image_width:=$WIDTH \
        -p image_height:=$HEIGHT \
        -p use_cuda:=true \
        -p wallclock:=$WALLCLOCK &
    PIDS+=($!)

    $DISPLAY_NODE --ros-args \
        -r __ns:=/cuda \
        -p use_cuda:=true \
        -p headless:=$HEADLESS \
        -p max_window_width:=$HALF_W \
        -p max_window_height:=$HALF_H \
        -p window_x:=0 \
        -p window_y:=0 &
    PIDS+=($!)

    echo "Starting CPU renderer + display (right window)..."
    $RENDERER --ros-args \
        -r __ns:=/cpu \
        -p image_width:=$WIDTH \
        -p image_height:=$HEIGHT \
        -p use_cuda:=false \
        -p wallclock:=$WALLCLOCK &
    PIDS+=($!)

    $DISPLAY_NODE --ros-args \
        -r __ns:=/cpu \
        -p use_cuda:=false \
        -p headless:=$HEADLESS \
        -p max_window_width:=$HALF_W \
        -p max_window_height:=$HALF_H \
        -p window_x:=$HALF_W \
        -p window_y:=0 &
    PIDS+=($!)

else
    echo "Starting renderer (publisher)..."
    echo "  Resolution: ${WIDTH}x${HEIGHT} ($RESOLUTION)"
    echo "  Backend: $BACKEND"
    echo "  Timestep: $TIMESTEP_STR"
    $RENDERER --ros-args \
        -p image_width:=$WIDTH \
        -p image_height:=$HEIGHT \
        -p use_cuda:=$USE_CUDA \
        -p wallclock:=$WALLCLOCK &
    PIDS+=($!)

    DISPLAY_ARGS="--ros-args -p use_cuda:=$USE_CUDA -p headless:=$HEADLESS"
    if [[ -n "$RECORD_PATH" ]]; then
        DISPLAY_ARGS="$DISPLAY_ARGS -p record_path:=$RECORD_PATH"
        echo "Starting display (subscriber, recording to $RECORD_PATH)..."
    else
        echo "Starting display (subscriber)..."
    fi
    $DISPLAY_NODE $DISPLAY_ARGS &
    PIDS+=($!)
fi

echo "All nodes running. Press Ctrl+C to stop."
wait
