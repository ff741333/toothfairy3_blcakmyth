$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DOCKER_TAG = "toothfairy3_blackmyth"

docker build $SCRIPT_DIR `
    --platform=linux/amd64 `
    --tag $DOCKER_TAG
