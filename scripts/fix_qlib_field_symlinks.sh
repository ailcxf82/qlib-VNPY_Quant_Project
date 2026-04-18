#!/usr/bin/env bash
# Create symlinks for missing standard Qlib fields.
# The data was prepared with _qfq (forward-adjusted) field names only.
# Qlib's PortAnaRecord and Exchange expect $close, $open, $high, $low to exist.
set -euo pipefail

DATA_DIR="${1:-/mnt/d/qlib_data/qlib_data/features}"

declare -A MAPPINGS=(
    ["close.day.bin"]="close_qfq.day.bin"
    ["open.day.bin"]="open_qfq.day.bin"
    ["high.day.bin"]="high_qfq.day.bin"
    ["low.day.bin"]="low_qfq.day.bin"
    ["volume.day.bin"]="vol.day.bin"
)

created=0
skipped=0
missing_src=0

for instrument_dir in "$DATA_DIR"/*/; do
    for target in "${!MAPPINGS[@]}"; do
        source="${MAPPINGS[$target]}"
        src_path="${instrument_dir}${source}"
        tgt_path="${instrument_dir}${target}"

        if [ -e "$tgt_path" ] || [ -L "$tgt_path" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        if [ ! -f "$src_path" ]; then
            missing_src=$((missing_src + 1))
            continue
        fi

        ln -s "$source" "$tgt_path"
        created=$((created + 1))
    done
done

echo "Done: created=$created skipped=$skipped missing_source=$missing_src"
