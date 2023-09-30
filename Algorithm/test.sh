#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# ./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="30g"

docker volume create lnq2023-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run \
	--gpus 0 \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v lnq2023-output-$VOLUME_SUFFIX:/output/ \
        lnq2023


#docker run --rm \
#        -v lnq2023-output-$VOLUME_SUFFIX:/output/ \
#        python:3.9-slim cat /output/results.json | python -m json.tool

#docker run --rm \
#        -v lnq2023-output-$VOLUME_SUFFIX:/output/ \
#        -v $SCRIPTPATH/test/:/input/ \
#        python:3.9-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"


if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm lnq2023-output-$VOLUME_SUFFIX
