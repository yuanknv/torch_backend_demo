#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="${PIXI_PROJECT_ROOT:-$SCRIPT_DIR}"

. "$WS_ROOT/build/tunnel_demo/colcon_command_prefix_build.sh"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export LD_LIBRARY_PATH="$WS_ROOT/build/tunnel_demo:$WS_ROOT/install/tunnel_demo/lib:$WS_ROOT/.pixi/envs/default/libtorch/lib:${LD_LIBRARY_PATH:-}"

ZENOHD="$WS_ROOT/install/rmw_zenoh_cpp/lib/rmw_zenoh_cpp/rmw_zenohd"
RENDERER="$WS_ROOT/build/tunnel_demo/tunnel_renderer_node"
DISPLAY_NODE="$WS_ROOT/build/tunnel_demo/tunnel_display_node"

RUN_SECONDS=15

SCENES="moving_objects tunnel"
BACKENDS="cuda cpu"
RESOLUTIONS="1920x1080 2560x1440 6144x3160"

run_bench() {
    local scene=$1 backend=$2 width=$3 height=$4
    local use_cuda="true"
    [[ "$backend" == "cpu" ]] && use_cuda="false"

    local logfile
    logfile=$(mktemp /tmp/bench_display_XXXXXX.log)

    $ZENOHD > /dev/null 2>&1 &
    local zpid=$!
    sleep 1

    $RENDERER --ros-args \
        -p image_width:=$width \
        -p image_height:=$height \
        -p use_cuda:=$use_cuda \
        -p publish_rate_ms:=1 \
        -p scene:=$scene > /dev/null 2>&1 &
    local rpid=$!

    $DISPLAY_NODE --ros-args -p headless:=true > "$logfile" 2>&1 &
    local dpid=$!

    sleep $RUN_SECONDS

    kill -INT $dpid $rpid 2>/dev/null || true
    wait $dpid $rpid 2>/dev/null || true
    kill $zpid 2>/dev/null || true
    wait $zpid 2>/dev/null || true

    local last_line
    last_line=$(grep "Display:" "$logfile" | tail -1)
    local fps latency
    fps=$(echo "$last_line" | grep -oP '[\d.]+(?= fps)')
    latency=$(echo "$last_line" | grep -oP 'latency: \K[\d.]+')

    echo "$scene|${width}x${height}|$backend|$fps|$latency"
    rm -f "$logfile"
    sleep 2
}

echo "scene|resolution|transport|fps|latency_ms"
for scene in $SCENES; do
    for res in $RESOLUTIONS; do
        width=${res%x*}
        height=${res#*x}
        for backend in $BACKENDS; do
            run_bench "$scene" "$backend" "$width" "$height"
        done
    done
done
