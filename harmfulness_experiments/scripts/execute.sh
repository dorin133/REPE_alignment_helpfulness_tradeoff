#!/bin/bash

ENV_NAME=fastdraft
SCRIPT=
TAIL=
USER_HOME=/home/ofirzafr

while test $# -gt 0; do
    case "$1" in
        --env)
            shift
            ENV_NAME=$1
            shift
            ;;
        --script)
            shift
            SCRIPT=$1
            shift
            ;;
        --user_home)
            shift
            USER_HOME=$1
            shift
            ;;
        *)
            TAIL+=' '$1
            shift
            ;;
    esac
done

CONDA_ROOT=$USER_HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_NAME

$SCRIPT $TAIL