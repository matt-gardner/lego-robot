#!/usr/bin/env bash
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# == 1 && "$1" =~ ^vol-[0-9a-f]+$ ]] || fail "Pass the newly provisioned EBS volume ID"
[[ "$(uname -s)" == Linux && $EUID == 0 ]] || fail "Requires Linux and root"
volume_id=$(printf '%s' "$1" | tr -d '-')
mount=/mnt/dockyard

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends docker.io containerd curl ca-certificates
systemctl stop docker.service docker.socket containerd.service

device=
for attempt in $(seq 1 120); do
    device=$(lsblk -dn -o NAME,SERIAL | awk -v serial="$volume_id" '$2 == serial {print "/dev/" $1}')
    [[ -n "$device" ]] && break
    sleep 5
done
[[ -n "$device" && -b "$device" ]] || fail "Expected EBS data volume did not attach"
[[ "$(printf '%s\n' "$device" | wc -l)" -eq 1 ]] || fail "Ambiguous data device"
root_device=$(findmnt -n -o SOURCE /)
if lsblk -snp -o NAME "$root_device" | grep -Fxq "$device"; then
    fail "Refusing to alter the OS disk"
fi
[[ "$(lsblk -bdn -o SIZE "$device")" -eq 85899345920 ]] || fail "Expected the provisioned 80 GiB data disk"
if mountpoint -q "$mount"; then
    [[ "$(findmnt -n -o MAJ:MIN --target "$mount")" == "$(lsblk -dn -o MAJ:MIN "$device")" ]] \
        || fail "Unexpected filesystem mounted at $mount"
else
    [[ -z "$(lsblk -nr -o MOUNTPOINTS "$device" | tr -d '[:space:]')" ]] \
        || fail "Data disk mounted elsewhere"
    if blkid -s TYPE -o value "$device" >/dev/null; then
        [[ "$(blkid -s TYPE -o value "$device")" == ext4 ]] || fail "Unexpected data filesystem"
    else
        [[ "$(lsblk -nr -o NAME "$device" | wc -l)" -eq 1 ]] || fail "Refusing to format a partitioned disk"
        [[ -z "$(wipefs -n --noheadings "$device")" ]] || fail "Refusing to erase existing signatures"
        mkfs.ext4 -L dockyard-data "$device"
    fi
    uuid=$(blkid -s UUID -o value "$device")
    mkdir -p "$mount"
    if ! grep -qF "UUID=$uuid " /etc/fstab; then
        printf 'UUID=%s %s ext4 defaults,nofail,x-systemd.device-timeout=120s 0 2\n' "$uuid" "$mount" >> /etc/fstab
    fi
    mount "$mount"
fi
mountpoint -q "$mount" || fail "Data mount is unavailable"
mkdir -p "$mount/docker" "$mount/containerd" "$mount/sessions"

mkdir -p /etc/docker /etc/containerd
cat > /etc/docker/daemon.json <<'EOF'
{
  "data-root": "/mnt/dockyard/docker",
  "cgroup-parent": "dockyard.slice",
  "exec-opts": ["native.cgroupdriver=systemd"],
  "log-driver": "local",
  "log-opts": {"max-size": "10m", "max-file": "3"}
}
EOF
containerd config default > /etc/containerd/config.toml
sed -i 's#^root = .*#root = "/mnt/dockyard/containerd"#' /etc/containerd/config.toml

cat > /etc/systemd/system/dockyard.slice <<'EOF'
[Unit]
Description=Dockyard container memory pool
[Slice]
MemoryAccounting=yes
MemoryMax=12G
EOF
for service in docker containerd; do
    mkdir -p "/etc/systemd/system/$service.service.d"
    cat > "/etc/systemd/system/$service.service.d/data-mount.conf" <<'EOF'
[Unit]
RequiresMountsFor=/mnt/dockyard
ConditionPathIsMountPoint=/mnt/dockyard
EOF
done
cat > /usr/local/sbin/dockyard-container-network <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if ! iptables -C DOCKER-USER -d 169.254.169.254/32 -j REJECT 2>/dev/null; then
    iptables -I DOCKER-USER 1 -d 169.254.169.254/32 -j REJECT
fi
EOF
chmod 755 /usr/local/sbin/dockyard-container-network
cat > /etc/systemd/system/docker.service.d/container-network.conf <<'EOF'
[Service]
ExecStartPost=/usr/local/sbin/dockyard-container-network
EOF
systemctl daemon-reload
systemctl enable containerd.service docker.service
systemctl start dockyard.slice containerd.service docker.service
docker info --format '{{.DockerRootDir}}' | grep -Fx /mnt/dockyard/docker
printf 'persistent-dockyard-data\n' > "$mount/persistence-check"
touch "$mount/.bootstrap-complete"
printf 'Dockyard CPU bootstrap complete; Docker and session state use the retained EBS disk.\n'
