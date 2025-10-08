#!/bin/bash

# need apt packages:  nodejs npm, the do 'sudo npm install -g http-server'

## display what's running on a given port: ss -lptn 'sport = :8081'
WORK_DIR=path-to-where-script-lives
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_DIR=$WORK_DIR/logs
LOG_PREFIX=$LOG_DIR/$TIMESTAMP

CONDA_ENV=name-of-conda-env-with-label-studio
SERVE_DIR=path-to-where-video-clips-live
SERVE_PORT=port-to-serve-clips

NPX_LOG="$LOG_PREFIX.npx.log"
LS_LOG="$LOG_PREFIX.label_studio.log"
NPX_PID_FILE="$LOG_DIR/npx.pid"
LS_PID_FILE="$LOG_DIR/label_studio.pid"

export LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true # no account creation without invitation link
# reverse proxy stuff
export LABEL_STUDIO_HOST=host-url-where-label-studio-lives # tells label studio where it lives

# CSRF and session cookies over HTTPS
export CSRF_TRUSTED_ORIGINS=host-url
export DJANGO_CSRF_COOKIE_SECURE=true
export DJANGO_CSRF_TRUSTED_ORIGINS=host-url

# Tell Django to trust proxy headers for host/port
export USE_X_FORWARDED_HOST=true
export USE_X_FORWARDED_PORT=true
export DJANGO_USE_X_FORWARDED_HOST=true
export DJANGO_USE_X_FORWARDED_PORT=true

# Ensure Django treats the connection as HTTPS if this header is set
export DJANGO_SECURE_PROXY_SSL_HEADER=HTTP_X_FORWARDED_PROTO,https

# Don't force Django to redirect HTTP→HTTPS (proxy handles HTTPS)
export DJANGO_SECURE_SSL_REDIRECT=false

# Hosts allowed in Host header
export DJANGO_ALLOWED_HOSTS=host-url,localhost,127.0.0.1


echo "label studio's new home is at: $LABEL_STUDIO_HOST"

echo "Usage: run_servers.sh start/stop"

# === HANDLE STOP COMMAND ===
if [ "$1" == "stop" ]; then
  echo "Stopping processes..."

  if [ -f "$NPX_PID_FILE" ]; then
    kill "$(cat "$NPX_PID_FILE")" && echo "Stopped npx http-server"
    rm "$NPX_PID_FILE"
  fi

  if [ -f "$LS_PID_FILE" ]; then
    kill "$(cat "$LS_PID_FILE")" && echo "Stopped Label Studio"
    rm "$LS_PID_FILE"
  fi

  exit 0
fi

if [ "$1" == "start" ]; then
    # === ACTIVATE CONDA ENVIRONMENT ===
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"

    # === START NPX HTTP-SERVER ===
    echo "Starting npx http-server..."

    nohup bash -c 'http-server "'"$SERVE_DIR"'" -p '"$SERVE_PORT"' --cors' > "$NPX_LOG" 2>&1 &
    echo $! > "$NPX_PID_FILE"

    # === START LABEL STUDIO ===
    echo "Starting Label Studio..."
    nohup bash -c label-studio --log-level DEBUG > "$LS_LOG" 2>&1 &
    echo $! > "$LS_PID_FILE"

    echo "Both services started."

fi


