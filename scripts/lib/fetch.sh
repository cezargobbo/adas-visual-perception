# Shared helpers for download_models.sh / download_videos.sh. Source, do not run.
# Exit codes: 0 ok, 1 download/checksum failure, 2 usage or missing dependency.

REPO_URL="https://github.com/cezargobbo/adas-visual-perception"
DEFAULT_TAG="v0.1.0"
SUMS_FILE=""

die() { # die CODE MESSAGE
  local code=$1
  shift
  echo "error: $*" >&2
  exit "$code"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die 2 "required command '$1' not found in PATH"
}

# fetch_sums BASE_URL: download SHA256SUMS into a temp file (path in SUMS_FILE).
fetch_sums() {
  SUMS_FILE=$(mktemp "${TMPDIR:-/tmp}/adasvp-sums.XXXXXX")
  trap 'rm -f "$SUMS_FILE"' EXIT
  curl -fsSL --retry 3 -o "$SUMS_FILE" "$1/SHA256SUMS" \
    || die 1 "could not download $1/SHA256SUMS"
}

# fetch_verified BASE_URL NAME DEST_DIR: download NAME into DEST_DIR, verified
# against SHA256SUMS. A valid existing file is kept; an invalid one is replaced.
# The final name only ever holds a verified file (download goes to NAME.part).
fetch_verified() {
  local base=$1 name=$2 dest=$3 want got
  want=$(awk -v n="$name" '$2 == n { print $1 }' "$SUMS_FILE")
  [ -n "$want" ] || die 1 "$name is not listed in SHA256SUMS"
  mkdir -p "$dest"
  if [ -f "$dest/$name" ]; then
    got=$(sha256sum "$dest/$name" | awk '{ print $1 }')
    if [ "$got" = "$want" ]; then
      echo "ok (already present): $dest/$name"
      return 0
    fi
    echo "existing $dest/$name has a bad checksum, downloading again"
  fi
  echo "downloading $name"
  rm -f "$dest/$name.part"
  if ! curl -fL --retry 3 -o "$dest/$name.part" "$base/$name"; then
    rm -f "$dest/$name.part"
    die 1 "download failed: $name"
  fi
  got=$(sha256sum "$dest/$name.part" | awk '{ print $1 }')
  if [ "$got" != "$want" ]; then
    rm -f "$dest/$name.part"
    die 1 "checksum mismatch for $name (expected $want, got $got)"
  fi
  mv -f "$dest/$name.part" "$dest/$name"
}
