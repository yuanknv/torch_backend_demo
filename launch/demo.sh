#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="${PIXI_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

resolve_resolution() {
    case "$1" in
        fhd|FHD|1080p) echo "1920 1080" ;;
        qhd|QHD|1440p) echo "2560 1440" ;;
        4k|4K|2160p)   echo "3840 2160" ;;
        *)
            echo "Error: unknown resolution '$1'. Use: fhd, qhd, 4k" >&2
            exit 1
            ;;
    esac
}

RESOLUTION="fhd"
BACKEND="cuda"
RECORD_PATH=""
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
            echo "  --resolution RES      fhd (default), qhd, 4k"
            echo "  --backend BACKEND     cuda or cpu (default: cuda)"
            echo "  --record PATH         Record video to MP4 file (requires ffmpeg)"
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

. "$WS_ROOT/install/setup.sh"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

RENDERER="$WS_ROOT/install/torch_backend_demo/lib/torch_backend_demo/renderer_node"
DISPLAY_NODE="$WS_ROOT/install/torch_backend_demo/lib/torch_backend_demo/display_node"

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

    if [[ "$COMPARE" == "true" && -n "$RECORD_PATH" ]]; then
        CUDA_REC="${RECORD_PATH%.mp4}_cuda.mp4"
        CPU_REC="${RECORD_PATH%.mp4}_cpu.mp4"
        if [[ -f "$CUDA_REC" && -f "$CPU_REC" ]]; then
            echo "Stitching recordings..."
            ffmpeg -y -i "$CUDA_REC" -i "$CPU_REC" \
                -filter_complex hstack "$RECORD_PATH" 2>/dev/null
            rm -f "$CUDA_REC" "$CPU_REC"
            echo "Saved: $RECORD_PATH"
        fi
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

if [[ "$COMPARE" == "true" ]]; then
    WIN_W=960
    WIN_H=540

    SCREEN_RES="$(xrandr 2>/dev/null | grep -oP '\d+x\d+\+0\+0' | head -1)"
    SCREEN_W="${SCREEN_RES%%x*}"
    SCREEN_H="${SCREEN_RES#*x}"; SCREEN_H="${SCREEN_H%%+*}"
    SCREEN_W="${SCREEN_W:-1920}"
    SCREEN_H="${SCREEN_H:-1080}"
    OFFSET_X=$(( (SCREEN_W - WIN_W * 2) / 2 ))
    OFFSET_Y=$(( (SCREEN_H - WIN_H) / 2 ))

    echo "=== Side-by-side comparison mode (two windows) ==="
    echo "  Resolution: ${WIDTH}x${HEIGHT} ($RESOLUTION)"
    echo "  Window: ${WIN_W}x${WIN_H} each, offset (${OFFSET_X}, ${OFFSET_Y})"

    $RENDERER --ros-args \
        -r __ns:=/cuda \
        -p image_width:=$WIDTH \
        -p image_height:=$HEIGHT \
        -p use_cuda:=true \
        --log-level WARN > /dev/null 2>&1 &
    PIDS+=($!)

    $RENDERER --ros-args \
        -r __ns:=/cpu \
        -p image_width:=$WIDTH \
        -p image_height:=$HEIGHT \
        -p use_cuda:=false \
        --log-level WARN > /dev/null 2>&1 &
    PIDS+=($!)

    CUDA_DISPLAY_ARGS="--ros-args -r __ns:=/cuda -p use_cuda:=true -p headless:=$HEADLESS"
    CPU_DISPLAY_ARGS="--ros-args -r __ns:=/cpu -p use_cuda:=false -p headless:=$HEADLESS"

    if [[ "$HEADLESS" == "false" ]]; then
        CPU_X=$((OFFSET_X + WIN_W))
        CUDA_DISPLAY_ARGS="$CUDA_DISPLAY_ARGS -p borderless:=true -p window_x:=$OFFSET_X -p window_y:=$OFFSET_Y"
        CUDA_DISPLAY_ARGS="$CUDA_DISPLAY_ARGS -p max_window_width:=$WIN_W -p max_window_height:=$WIN_H"
        CPU_DISPLAY_ARGS="$CPU_DISPLAY_ARGS -p borderless:=true -p window_x:=$CPU_X -p window_y:=$OFFSET_Y"
        CPU_DISPLAY_ARGS="$CPU_DISPLAY_ARGS -p max_window_width:=$WIN_W -p max_window_height:=$WIN_H"
    fi

    if [[ -n "$RECORD_PATH" ]]; then
        CUDA_REC="${RECORD_PATH%.mp4}_cuda.mp4"
        CPU_REC="${RECORD_PATH%.mp4}_cpu.mp4"
        CUDA_DISPLAY_ARGS="$CUDA_DISPLAY_ARGS -p record_path:=$CUDA_REC"
        CPU_DISPLAY_ARGS="$CPU_DISPLAY_ARGS -p record_path:=$CPU_REC"
        echo "  Recording to $RECORD_PATH (via stitch)"
    fi

    $DISPLAY_NODE $CUDA_DISPLAY_ARGS &
    PIDS+=($!)
    $DISPLAY_NODE $CPU_DISPLAY_ARGS &
    PIDS+=($!)

else
    echo "Starting renderer (publisher)..."
    echo "  Resolution: ${WIDTH}x${HEIGHT} ($RESOLUTION)"
    echo "  Backend: $BACKEND"
    $RENDERER --ros-args \
        -p image_width:=$WIDTH \
        -p image_height:=$HEIGHT \
        -p use_cuda:=$USE_CUDA &
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
