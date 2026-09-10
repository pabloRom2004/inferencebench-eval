#!/bin/bash
set -euo pipefail

# Preserve the installed filesystem independently of RunPod's container disk lifecycle.
state=/workspace/.inferencebench
exclude=(--exclude=./workspace --exclude=./proc --exclude=./sys --exclude=./dev
         --exclude=./run --exclude=./etc/hosts --exclude=./etc/hostname --exclude=./etc/resolv.conf)

# NVIDIA injects this mount independently of the installed container filesystem.
# Exclude its directory inventory and metadata as well as its mounted contents.
if python3 - <<'PYMOUNT'
from pathlib import Path

profile = '/etc/nvidia/nvidia-application-profiles-rc.d'
mounted = any(line.split()[4] == profile
              for line in Path('/proc/self/mountinfo').read_text().splitlines())
raise SystemExit(0 if mounted else 1)
PYMOUNT
then
    exclude+=(--exclude=./etc/nvidia/nvidia-application-profiles-rc.d)
fi

if [ "${1:-start}" = snapshot ]; then
    mkdir -p "$state"
    rm -f "$state/ready"
    # Freeze background writers; retain this command and its SSH ancestor chain.
    # The imminent restart discards stopped processes; snapshot errors trigger pod cleanup.
    python3 - <<'PY'
import os
import signal
import time
from pathlib import Path

ancestors = {1}
pid = os.getpid()
while pid > 1:
    ancestors.add(pid)
    fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
    pid = int(fields['PPid'])

while True:
    running = False
    for path in Path('/proc').glob('[0-9]*/status'):
        pid = int(path.parent.name)
        if pid in ancestors:
            continue
        try:
            fields = dict(line.split(':', 1) for line in path.read_text().splitlines())
            if fields['State'].strip()[0] not in 'TtZX':
                os.kill(pid, signal.SIGSTOP)
                running = True
        except (FileNotFoundError, ProcessLookupError):
            pass
    if not running:
        break
    time.sleep(0.01)
PY
    # One sequential archive avoids thousands of network-volume metadata round trips.
    tar --format=pax --incremental --numeric-owner --sparse --one-file-system \
        --blocking-factor=16384 "${exclude[@]}" \
        -C / -cf "$state/root.tar.partial" .
    mv "$state/root.tar.partial" "$state/root.tar"
    touch "$state/ready"
    exit 0
fi

if [ -f "$state/ready" ]; then
    # Directory inventories restore deletions; direct extraction needs no second copy.
    tar --incremental --numeric-owner "${exclude[@]}" \
        -xpf "$state/root.tar" -C /
fi

# SSH keys are per-pod; the host's account and model-provider keys are not copied here.
install -d -m 700 /root/.ssh
mkdir -p /run/sshd
printf '%s\n' "$INFERENCEBENCH_SSH_PUBLIC_KEY" > /root/.ssh/authorized_keys
printf '%s\n' "$INFERENCEBENCH_SSH_HOST_KEY" > /etc/ssh/ssh_host_ed25519_key
chmod 600 /root/.ssh/authorized_keys /etc/ssh/ssh_host_ed25519_key
ssh-keygen -y -f /etc/ssh/ssh_host_ed25519_key > /etc/ssh/ssh_host_ed25519_key.pub
unset INFERENCEBENCH_SSH_PUBLIC_KEY INFERENCEBENCH_SSH_HOST_KEY

# SSH commands need the same CUDA and model-cache environment as Modal commands.
export -p > /etc/inferencebench-env.sh
python3 -c 'import uuid; print(uuid.uuid4())' > /run/inferencebench-boot-id
exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no -o UsePAM=no \
    -o PermitRootLogin=prohibit-password -o AuthorizedKeysFile=/root/.ssh/authorized_keys
