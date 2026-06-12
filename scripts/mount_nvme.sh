#!/usr/bin/env bash
set -euo pipefail

: "${DATA_DEVICE:?DATA_DEVICE is required}"
: "${INDEX_DEVICE:?INDEX_DEVICE is required}"

if [[ "${DATA_DEVICE}" == "${INDEX_DEVICE}" ]]; then
  echo "DATA_DEVICE and INDEX_DEVICE must be different" >&2
  exit 2
fi

for device in "${DATA_DEVICE}" "${INDEX_DEVICE}"; do
  if [[ ! -b "${device}" ]]; then
    echo "Not a block device: ${device}" >&2
    exit 2
  fi
done

echo "This destroys all data on:"
echo "  ${DATA_DEVICE} -> /mnt/tv-data"
echo "  ${INDEX_DEVICE} -> /mnt/tv-index"
read -r -p "Type WIPE to continue: " confirmation
[[ "${confirmation}" == "WIPE" ]] || exit 2

umount /mnt/tv-data 2>/dev/null || true
umount /mnt/tv-index 2>/dev/null || true

wipefs -af "${DATA_DEVICE}"
wipefs -af "${INDEX_DEVICE}"
mkfs.xfs -f "${DATA_DEVICE}"
mkfs.xfs -f "${INDEX_DEVICE}"

mkdir -p /mnt/tv-data /mnt/tv-index
mount -o noatime,nodiratime "${DATA_DEVICE}" /mnt/tv-data
mount -o noatime,nodiratime "${INDEX_DEVICE}" /mnt/tv-index

mkdir -p /mnt/tv-data/msmarco /mnt/tv-index/indexes
chown -R "${SUDO_USER:-ubuntu}:${SUDO_USER:-ubuntu}" \
  /mnt/tv-data /mnt/tv-index

df -hT /mnt/tv-data /mnt/tv-index
