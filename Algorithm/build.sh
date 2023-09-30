#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t lnq2023 "$SCRIPTPATH" --no-cache
