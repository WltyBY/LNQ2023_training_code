#!/usr/bin/env bash

./build.sh

docker save lnq2023 | gzip -c > LNQ2023.tar.gz
