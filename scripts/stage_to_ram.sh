#!/usr/bin/env bash
# Stage the resized dataset into RAM (tmpfs) before training.
#
# Both disks on this machine are spinning platters behind a RAID controller,
# and /data is shared. The input pipeline issues small random reads across
# 1.28M files, which is close to the worst case for that storage: training sat
# at 44% iowait with one GPU idle, managing 1,016 img/s.
#
# The resized dataset is only ~38 GB and the box has ~500 GB of RAM, so it fits
# in /dev/shm with room to spare. Doing this took throughput to ~6,200 img/s --
# a 6x gain, far larger than anything the choice of parallelism strategy is
# worth here.
#
# tmpfs is volatile: the copy is lost on reboot, and it holds RAM until removed
# with `rm -rf "$DST"`.
set -euo pipefail

SRC=${SRC:-/data/users/cs24s008/Datasets/ilsvrc2012_256}
DST=${DST:-/dev/shm/ilsvrc2012_256}

need_gb=$(du -sBG --apparent-size "$SRC" | cut -f1 | tr -d 'G')
free_gb=$(df -BG --output=avail /dev/shm | tail -1 | tr -d 'G ')
echo "dataset ${need_gb}G, /dev/shm free ${free_gb}G"
if (( free_gb < need_gb + 5 )); then
    echo "not enough space in /dev/shm" >&2
    exit 1
fi

mkdir -p "$DST"
cp -r "$SRC/." "$DST/"
echo "staged $(find "$DST" -type f | wc -l) files to $DST"
echo "train with:  --set data.root=$DST"
