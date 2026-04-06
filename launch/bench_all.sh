#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="${PIXI_PROJECT_ROOT:-$SCRIPT_DIR}"

. "$WS_ROOT/build/torch_backend_demo/colcon_command_prefix_build.sh"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$WS_ROOT/build/torch_backend_demo:$WS_ROOT/install/torch_backend_demo/lib:$WS_ROOT/.pixi/envs/default/libtorch/lib:${LD_LIBRARY_PATH:-}"

ZENOHD="$WS_ROOT/install/rmw_zenoh_cpp/lib/rmw_zenoh_cpp/rmw_zenohd"
RENDERER="$WS_ROOT/build/torch_backend_demo/renderer_node"
DISPLAY_NODE="$WS_ROOT/build/torch_backend_demo/display_node"

RUN_SECONDS=15

BACKENDS="cuda cpu"
PRESETS="fhd qhd 4k"

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

run_bench() {
    local backend=$1 preset=$2
    read -r width height <<< "$(resolve_resolution "$preset")"
    local use_cuda="true"
    [[ "$backend" == "cpu" ]] && use_cuda="false"

    local logfile
    logfile=$(mktemp /tmp/bench_display_XXXXXX.log)

    ZENOH_PID=""

    $RENDERER --ros-args \
        -p image_width:=$width \
        -p image_height:=$height \
        -p use_cuda:=$use_cuda > /dev/null 2>&1 &
    local rpid=$!

    $DISPLAY_NODE --ros-args -p headless:=true > "$logfile" 2>&1 &
    local dpid=$!

    sleep $RUN_SECONDS

    kill -INT $dpid $rpid 2>/dev/null || true
    wait $dpid $rpid 2>/dev/null || true
    if [[ -n "$ZENOH_PID" ]]; then
        kill $ZENOH_PID 2>/dev/null || true
        wait $ZENOH_PID 2>/dev/null || true
    fi

    local last_line
    last_line=$(grep "Display:" "$logfile" | tail -1)
    local fps
    fps=$(echo "$last_line" | grep -oP '[\d.]+(?= fps)')

    echo "${width}x${height}|$backend|$fps"
    rm -f "$logfile"
    sleep 2
}

echo "resolution|transport|fps"
for preset in $PRESETS; do
    for backend in $BACKENDS; do
        run_bench "$backend" "$preset"
    done
done
