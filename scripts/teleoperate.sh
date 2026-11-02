#!/usr/bin/env bash
set -euo pipefail

: "${FOLLOWER_PORT:=/dev/tty.usbmodem5B901012531}"
: "${LEADER_PORT:=/dev/tty.usbmodem5B901006131}"
: "${CAMERA_INDEX:=0}"

if [[ "$FOLLOWER_PORT" == "$LEADER_PORT" ]]; then
    printf '%s\n' 'FOLLOWER_PORT and LEADER_PORT must be different.' >&2
    exit 1
fi
if [[ ! "$CAMERA_INDEX" =~ ^[0-9]+$ ]]; then
    printf '%s\n' 'CAMERA_INDEX must be a nonnegative integer.' >&2
    exit 1
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

exec uv run --locked lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id=so101_follower \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id=so101_leader \
    --robot.cameras="{front: {type: opencv, index_or_path: $CAMERA_INDEX, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --fps=30 \
    "$@"
