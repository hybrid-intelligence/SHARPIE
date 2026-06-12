#!/bin/bash
set -e  # Exit on error (but continue to next use-case if one fails)

# Configuration
SHARPIE_DIR="/var/www/sharpie"
WEBSERVER_DIR="$SHARPIE_DIR/webserver"
RUNNER_DIR="$SHARPIE_DIR/runner"
GALLERY_REPO="https://github.com/hybrid-intelligence/SHARPIE_Gallery.git"
TEMP_DIR="/tmp/SHARPIE_Gallery_$$"  # Unique temp directory
USE_CASES_FILE="$SHARPIE_DIR/deployment/use_cases.txt"
LOG_FILE="$SHARPIE_DIR/logs/use_cases_install.log"

# Ensure logs directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Clone SHARPIE_Gallery
log "Cloning SHARPIE_Gallery..."
git clone --depth 1 "$GALLERY_REPO" "$TEMP_DIR"

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
    
    # Check if use-case directory exists in Gallery
    if [ ! -d "$TEMP_DIR/$use_case" ]; then
        log "✗ Use-case not found: $use_case"
        FAILED_USE_CASES="$FAILED_USE_CASES $use_case"
        continue
    fi
    
    # Copy use-case files to runner directory
    # Each use-case has environment.py, policy.py, etc. in its directory
    # These need to be accessible from the runner
    if [ -d "$TEMP_DIR/$use_case" ]; then
        log "  Copying files to runner directory..."
        mkdir -p "$RUNNER_DIR/$use_case"
        cp -r "$TEMP_DIR/$use_case"/* "$RUNNER_DIR/$use_case/"
        log "  ✓ Files copied to: $RUNNER_DIR/$use_case"
    fi
    
    # Install dependencies and update database
    OUTPUT=$(python "$TEMP_DIR/install.py" "$use_case" \
         --sharpie-dir "$SHARPIE_DIR" \
         --webserver-dir "$WEBSERVER_DIR" \
         --quiet 2>&1)
    EXIT_CODE=$?
    echo "$OUTPUT" | tee -a "$LOG_FILE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "✓ Successfully installed: $use_case"
    else
        log "✗ Failed to install: $use_case"
        FAILED_USE_CASES="$FAILED_USE_CASES $use_case"
        # Clean up copied files if installation failed
        rm -rf "$RUNNER_DIR/$use_case"
    fi
done

# Cleanup
log "Cleaning up..."
rm -rf "$TEMP_DIR"

# Summary
if [ -n "$FAILED_USE_CASES" ]; then
    log "WARNING: Some use-cases failed to install:$FAILED_USE_CASES"
    log "Deployment continuing with successful installations."
    # Don't exit with error - allow deployment to continue
fi

log "Use-case installation complete."