#!/bin/bash
set -e

# Configuration
SHARPIE_DIR="/var/www/sharpie"
GALLERY_REPO="https://github.com/hybrid-intelligence/SHARPIE_Gallery.git"
GALLERY_DIR="$(dirname "$SHARPIE_DIR")/SHARPIE_Gallery"
USE_CASES_FILE="$SHARPIE_DIR/deployment/use_cases.txt"
LOG_FILE="$SHARPIE_DIR/logs/use_cases_install.log"

# Ensure logs directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Ensure we run from the webserver directory (same CWD as sharpie-web migrate)
# so that Django resolves db.sqlite3 to the correct path
cd "$SHARPIE_DIR/webserver"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Clone or update SHARPIE_Gallery
if [ -d "$GALLERY_DIR" ]; then
    log "Updating existing SHARPIE_Gallery..."
    git -C "$GALLERY_DIR" fetch origin
    git -C "$GALLERY_DIR" reset --hard origin/main
else
    log "Cloning SHARPIE_Gallery..."
    git clone --depth 1 "$GALLERY_REPO" "$GALLERY_DIR"
fi

# Read use-cases from config file
USE_CASES=$(grep -v '^#' "$USE_CASES_FILE" | grep -v '^$' || true)

if [ -z "$USE_CASES" ]; then
    log "ERROR: No use-cases found in $USE_CASES_FILE"
    exit 1
fi

# Install each use-case
FAILED_USE_CASES=""
for use_case in $USE_CASES; do
    log "Installing use-case: $use_case"
    
    set +e
    sharpie-install "$use_case" --gallery-dir "$GALLERY_DIR" --quiet 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=$?
    set -e
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "✓ Successfully installed: $use_case"
    else
        log "✗ Failed to install: $use_case"
        FAILED_USE_CASES="$FAILED_USE_CASES $use_case"
    fi
done

# Summary
if [ -n "$FAILED_USE_CASES" ]; then
    log "WARNING: Some use-cases failed to install:$FAILED_USE_CASES"
    log "Deployment continuing with successful installations."
fi

log "Use-case installation complete."