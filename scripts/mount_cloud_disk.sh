#!/usr/bin/env bash
set -euo pipefail

# Only LUN 0 of the separately provisioned Azure data disk is in scope.
device=/dev/disk/azure/scsi1/lun0
mount=/mnt/training
initialize=0
if [[ "${1:-}" == --initialize-empty-disk && $# == 1 ]]; then
    initialize=1
elif (($#)); then
    printf 'Usage: sudo bash scripts/mount_cloud_disk.sh [--initialize-empty-disk]\n' >&2
    exit 1
fi
[[ $EUID == 0 ]] || { echo "Run using sudo" >&2; exit 1; }
[[ -b "$device" ]] || { echo "Azure managed data disk LUN 0 is missing" >&2; exit 1; }
device=$(readlink -f "$device")
root_device=$(findmnt -n -o SOURCE /)
[[ "$device" != "$root_device" ]] || exit 1
if lsblk -snp -o NAME "$root_device" | grep -Fxq "$device"; then
    echo "Refusing to alter the OS disk" >&2
    exit 1
fi
if mountpoint -q "$mount"; then
    [[ "$(findmnt -n -o MAJ:MIN --target "$mount")" == "$(lsblk -dn -o MAJ:MIN "$device")" ]] \
        || { echo "Unexpected filesystem already mounted at $mount" >&2; exit 1; }
    exit 0
fi
[[ -z "$(lsblk -nr -o MOUNTPOINTS "$device" | tr -d '[:space:]')" ]] \
    || { echo "Data disk is already mounted elsewhere" >&2; exit 1; }
if [[ -z "$(blkid -s TYPE -o value "$device" || true)" ]]; then
    ((initialize)) || { echo "Blank disk: explicitly pass --initialize-empty-disk once" >&2; exit 1; }
    [[ "$(lsblk -nr -o NAME "$device" | wc -l)" -eq 1 ]] \
        || { echo "Refusing to format a partitioned disk" >&2; exit 1; }
    [[ -z "$(wipefs -n --noheadings "$device")" ]] \
        || { echo "Refusing to format a disk with existing signatures" >&2; exit 1; }
    mkfs.ext4 -L so101-training "$device"
fi
[[ "$(blkid -s TYPE -o value "$device")" == ext4 ]] \
    || { echo "Expected an ext4 data disk; refusing to alter it" >&2; exit 1; }
uuid=$(blkid -s UUID -o value "$device")
mkdir -p "$mount"
grep -qF "UUID=$uuid " /etc/fstab \
    || printf 'UUID=%s %s ext4 defaults,nofail 0 2\n' "$uuid" "$mount" >> /etc/fstab
mount "$mount"
chown "${SUDO_USER:?Run through sudo}:$(id -gn "$SUDO_USER")" "$mount"
