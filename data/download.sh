#!/usr/bin/env bash
#
# SAMI-Audio — NSynth Subset Download Script
# ===========================================
# Scarica il dataset NSynth train set (~21 GB), estrae i file audio,
# filtra per le 4 famiglie strumentali e il pitch range del progetto,
# ed elimina i file .wav non necessari per risparmiare spazio.
#
# Usage:
#   bash data/download.sh          # download + extract + filter (interattivo)
#   bash data/download.sh --yes    # salta le conferme (non interattivo)
#
# Requisiti: curl, tar, Python 3.10+
#
# Dopo l'esecuzione, la struttura sarà:
#   data/nsynth-train/
#   ├── examples.json          # metadati completi
#   └── audio/
#       ├── guitar_acoustic_0...wav
#       ├── keyboard_electronic_0...wav
#       └── ...                 # solo i file delle 4 famiglie + pitch 48-84
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}"
NSYNTH_URL="http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
TAR_FILE="${DATA_DIR}/nsynth-train.jsonwav.tar.gz"
EXTRACT_DIR="${DATA_DIR}/nsynth-train"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

AUTO_YES=false
if [[ "${1:-}" == "--yes" ]]; then
    AUTO_YES=true
fi

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

confirm() {
    if $AUTO_YES; then
        return 0
    fi
    local prompt="$1"
    read -r -p "$(echo -e "${YELLOW}[?]${NC} ${prompt} [y/N] ")" response
    case "$response" in
        [yY][eE][sS]|[yY]) return 0 ;;
        *) return 1 ;;
    esac
}

echo ""
echo "============================================"
echo "  SAMI-Audio — NSynth Subset Downloader"
echo "============================================"
echo ""
log_info "Data directory : ${DATA_DIR}"
log_info "Extract target : ${EXTRACT_DIR}"
log_info "Download URL  : ${NSYNTH_URL}"
log_info "Expected size : ~21 GB (compressed), ~4 GB after filtering"
echo ""

# --- Step 1: Download ----------------------------------------------------------
log_info "Step 1/4 — Downloading NSynth train set..."

if [[ -f "${TAR_FILE}" ]]; then
    existing_size=$(stat -f%z "${TAR_FILE}" 2>/dev/null || stat -c%s "${TAR_FILE}" 2>/dev/null || echo 0)
    log_warn "Tar file already exists (${existing_size} bytes)."
    if confirm "Resume download or skip?"; then
        log_info "Resuming download (curl -C -)..."
    else
        log_info "Skipping download."
    fi
else
    log_info "Starting download (~21 GB — this will take a while)..."
fi

if ! curl -L -C - --progress-bar -o "${TAR_FILE}" "${NSYNTH_URL}"; then
    log_error "Download failed. Check your internet connection and retry."
    log_info "You can resume the download by running this script again."
    exit 1
fi

TAR_SIZE=$(stat -f%z "${TAR_FILE}" 2>/dev/null || stat -c%s "${TAR_FILE}" 2>/dev/null || echo 0)
log_ok "Download complete (${TAR_SIZE} bytes)"

# --- Step 2: Extract ------------------------------------------------------------
log_info "Step 2/4 — Extracting archive..."

if [[ -d "${EXTRACT_DIR}" ]] && [[ -f "${EXTRACT_DIR}/examples.json" ]]; then
    log_warn "Extraction directory already exists with examples.json."
    if confirm "Skip extraction?"; then
        log_info "Skipping extraction."
    else
        log_info "Re-extracting..."
        tar -xzf "${TAR_FILE}" -C "${DATA_DIR}"
    fi
else
    log_info "Extracting nsynth-train.jsonwav.tar.gz ..."
    tar -xzf "${TAR_FILE}" -C "${DATA_DIR}"
fi

if [[ ! -f "${EXTRACT_DIR}/examples.json" ]]; then
    log_error "Extraction failed: examples.json not found in ${EXTRACT_DIR}"
    exit 1
fi

TOTAL_WAVS=$(find "${EXTRACT_DIR}/audio" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
log_ok "Extraction complete — ${TOTAL_WAVS} .wav files found"

# --- Step 3: Filter -------------------------------------------------------------
log_info "Step 3/4 — Filtering dataset (4 families, pitch 48-84)..."

python3 << 'PYEOF'
import json
import os
import sys

extract_dir = os.environ.get("EXTRACT_DIR", os.path.join(os.path.dirname(__file__), "nsynth-train"))
json_path = os.path.join(extract_dir, "examples.json")
audio_dir = os.path.join(extract_dir, "audio")

TARGET_FAMILIES = {"guitar", "keyboard", "string", "brass"}
PITCH_MIN, PITCH_MAX = 48, 84

print(f"[INFO]  Loading metadata from {json_path}...")
with open(json_path, "r") as f:
    data = json.load(f)

total = len(data)
print(f"[INFO]  Total entries in dataset: {total}")

matching_filenames = set()
family_counts = {fam: 0 for fam in TARGET_FAMILIES}
pitch_counts = {}
skipped_family = 0
skipped_pitch = 0
skipped_source = 0

for entry_id, entry in data.items():
    family = entry.get("instrument_family_str", entry.get("instrument_family", ""))
    if isinstance(family, int):
        family_map = {
            0: "bass", 1: "brass", 2: "flute", 3: "guitar",
            4: "keyboard", 5: "mallet", 6: "organ", 7: "reed",
            8: "string", 9: "synth_lead", 10: "vocal"
        }
        family = family_map.get(family, str(family))
    family = str(family).lower()

    pitch = entry.get("pitch", entry.get("note", -1))
    if isinstance(pitch, str):
        try:
            pitch = int(pitch)
        except ValueError:
            pitch = -1

    if family not in TARGET_FAMILIES:
        skipped_family += 1
        continue

    if not (PITCH_MIN <= pitch <= PITCH_MAX):
        skipped_pitch += 1
        continue

    matching_filenames.add(f"{entry_id}.wav")
    family_counts[family] += 1
    pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1

print(f"\n[INFO]  Filter results:")
print(f"  Kept:   {len(matching_filenames)} samples")
print(f"  Skipped (wrong family): {skipped_family}")
print(f"  Skipped (wrong pitch):  {skipped_pitch}")
print(f"\n  By family:")
for fam, count in sorted(family_counts.items()):
    print(f"    {fam:<12} {count:>6}")
print(f"\n  Pitch range: {min(pitch_counts.keys()) if pitch_counts else 'N/A'} - "
      f"{max(pitch_counts.keys()) if pitch_counts else 'N/A'} "
      f"(unique: {len(pitch_counts)})")

# Delete non-matching wav files
wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
deleted = 0
for wav in wav_files:
    if wav not in matching_filenames:
        os.remove(os.path.join(audio_dir, wav))
        deleted += 1

remaining = len([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
print(f"\n[INFO]  Deleted {deleted} non-matching .wav files")
print(f"[INFO]  Remaining: {remaining} .wav files")
assert remaining == len(matching_filenames), \
    f"Mismatch: {remaining} files on disk vs {len(matching_filenames)} in index"
print(f"[OK]    File count matches metadata entries ({remaining})")
PYEOF

FILTER_EXIT_CODE=$?
if [[ $FILTER_EXIT_CODE -ne 0 ]]; then
    log_error "Filtering failed with exit code ${FILTER_EXIT_CODE}"
    exit 1
fi

REMAINING=$(find "${EXTRACT_DIR}/audio" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
log_ok "Filtering complete — ${REMAINING} .wav files remain"

# --- Step 4: Cleanup ------------------------------------------------------------
log_info "Step 4/4 — Cleanup..."
if [[ -f "${TAR_FILE}" ]]; then
    if confirm "Delete the downloaded tar.gz (~21 GB) to free space?"; then
        rm -f "${TAR_FILE}"
        log_ok "Deleted ${TAR_FILE}"
    else
        log_info "Keeping ${TAR_FILE}. You can delete it manually later."
    fi
fi

echo ""
echo "============================================"
log_ok "NSynth subset ready!"
echo ""
echo "  Directory : ${EXTRACT_DIR}"
echo "  Samples   : ${REMAINING}"
echo "  Families  : guitar, keyboard, string, brass"
echo "  Pitch     : 48-84 (C3-B5)"
echo ""
echo "  Next step: pdm run gate0"
echo "============================================"
echo ""
