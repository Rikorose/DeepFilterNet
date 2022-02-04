#!/bin/bash
#SBATCH --job-name=df
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --gres=gpu:1
#SBATCH -o /cluster/%u/logs/%j.out
#SBATCH -e /cluster/%u/logs/%j.out
#SBATCH --mail-type=ALL
# Time limit format: "hours:minutes:seconds"
#SBATCH --time=24:00:00
# Send the SIGUSR1 signal 2 h before time limit
#SBATCH --signal=B:USR1@7200

set -e

# Cluster directory containing data, project code, logging, summaries,
# checkpoitns, etc.
export CLUSTER=/cluster/$USER
cd "$CLUSTER"

# Workaround for our cluster
export WORKON_HOME="/cluster/$USER/.cache"
export XDG_CACHE_DIR="/cluster/$USER/.cache"
export PYTHONUSERBASE="/cluster/$USER/.python_packages"
# Workaround for HDF5 on other file systems. All hdf5 files are opened in read
# only mode; we do not need locks.
export HDF5_USE_FILE_LOCKING='FALSE'

PROJECT_NAME=DeepFilterNet
DATA_DIR=${DATA_DIR:-$CLUSTER/Data/HDF5}     # Set to the directory containing the HDF5s
DATA_CFG=${DATA_CFG:-$DATA_DIR/datasets.cfg} # Default dataset configuration
DATA_CFG=$(readlink -f "$DATA_CFG")
PYTORCH_JIT=${PYTORCH_JIT:-1}                # Set to 0 to disable pytorch JIT compilation
COPY_DATA=${COPY_DATA:-1}                    # Copy data
DEBUG=${DEBUG:-0}                            # Debug mode passed to the python train script
EXCLUDE=${EXCLUDE:-lme[49,170,171]}          # Slurm nodes to exclude

if [ "$DEBUG" -eq 1 ]; then
  DEBUG="--debug"
else
  DEBUG=
fi

echo "Started sbatch script at $(date) in $(pwd)"

echo "Found cuda devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L || echo "nvidia-smi not found"
echo "Running on host: $(hostname)"

# Check base dir file
if [[ -z $1 ]]; then
  echo >&2 "No model base directory provided!"
  exit
fi
if [[ ! -d $1 ]]; then
  echo >&2 "Model base directory not found at $1!"
  exit
fi

BASE_DIR=$(readlink -f "$1")

echo "Got base_dir: $BASE_DIR"
MODEL_NAME=$(basename "$BASE_DIR")

# Git project setup.
# Creates a separate code directory so that changes of the source code don't
# have any impact on the automatic resubmission process. Furthermore, a specific
# branch or commit can be specified that will be tried to checkout. By default,
# the currently active branch is used.
PROJECT_BRANCH=${BRANCH:-$PROJECT_BRANCH_CUR}
if [[ -n $2 ]]; then
  if [[ ! -d $2 ]]; then
    echo >&2 "Project home not found at $2!"
    exit
  fi
  PROJECT_HOME=$2
else
  PROJECT_ClUSTER_HOME=$CLUSTER/$PROJECT_NAME/
  PROJECT_HOME=$CLUSTER/sbatch-$PROJECT_NAME/$MODEL_NAME/
  mkdir -p "$PROJECT_HOME"
  echo "Copying repo to $PROJECT_HOME"
  cd "$PROJECT_ClUSTER_HOME"
  rsync -avq --include .git \
    --exclude-from="$(git -C "$PROJECT_ClUSTER_HOME" ls-files --exclude-standard -oi --directory >.git/ignores.tmp && echo .git/ignores.tmp)" \
    "$PROJECT_ClUSTER_HOME" "$PROJECT_HOME" --delete
fi
if [ -n "$3" ]; then
  # Checkout specified branch from previous job
  PROJECT_BRANCH_CUR=$3
else
  # Use current branch of project on /cluster
  PROJECT_BRANCH_CUR=$(git -C "$PROJECT_ClUSTER_HOME" rev-parse --abbrev-ref HEAD)
fi
echo "Running on branch $PROJECT_BRANCH in dir $PROJECT_HOME"
if [ "$PROJECT_BRANCH_CUR" != "$PROJECT_BRANCH" ]; then
  stash="stash_$SLURM_JOB_ID"
  git -C "$PROJECT_HOME" stash save "$stash"
  git -C "$PROJECT_HOME" checkout "$PROJECT_BRANCH"
  stash_idx=$(git -C "$PROJECT_HOME" stash list | grep "$stash" | cut -d: -f1)
  if [ ! -z "$stash_idx" -a "$stash_idx" != " " ]; then
    # Try to apply current stash; If not possible just proceed.
    if ! git -C "$PROJECT_HOME" stash pop "$stash_idx"; then
      echo "Could not apply stash to branch $PROJECT_BRANCH"
      git -C "$PROJECT_HOME" checkout -f
    fi
  fi
fi

# Setup conda environment.
# This installs miniconda environment if not existing, pytorch with cuda
# integration and pip packages in requirements.txt
. "$PROJECT_HOME"/scripts/setup_env.sh --source-only
setup_env "$CLUSTER" "$PROJECT_HOME" "$MODEL_NAME"

# Copy data from shared file system to a local folder
if [[ -d /scratch ]] && [[ $COPY_DATA -eq 1 ]]; then
  test -d "/scratch/$USER" || mkdir "/scratch/$USER"
  NEW_DATA_DIR=/scratch/"$USER"/"$PROJECT_NAME"
  echo "Seting up data dir in $NEW_DATA_DIR"
  mkdir -p "$NEW_DATA_DIR"
  # Check if another process is currently copying
  while true; do
    LOCK=$(rg lock "$NEW_DATA_DIR"/data_lock | sed "s/-lock//g")
    if [[ -n $LOCK ]]; then
      echo "Scratch dir is currently locked by: $LOCK"
      sleep 30s
    else
      break
    fi
  done
  echo "$MODEL_NAME-lock" >> "$NEW_DATA_DIR"/data_lock  # lock scratch dir
  MAX_GB=150 "$PROJECT_HOME"/scripts/copy_datadir.sh "$DATA_DIR" "$DATA_CFG" "$NEW_DATA_DIR"
  # release copy lock; remaining $MODEL_NAME indicates we are using it
  sed -i "s/$MODEL_NAME-lock/$MODEL_NAME/g"  "$NEW_DATA_DIR"/data_lock
  DATA_DIR="$NEW_DATA_DIR"
fi

# Signal handlers.
# This is used to indicate that maximum training time will be exceeded.
# Therefore, we send a SIGUSR1 to the python training script which needs to
# write a continue file if it wants to get restarted. After training scripts
# returns, check if there is a continue file and resubmit.
function _at_exit {
  conda deactivate
  # Check for return code if training was completed
  echo "Checking if need to resubmit training script"
  python3 "$PROJECT_HOME"/scripts/has_continue_file.py "$BASE_DIR"
  retVal=$?
  if [ $retVal -eq 0 ]; then
    echo "Training not completed. Resubmitting to continue training."
    sh -c "sbatch --exclude=$EXCLUDE \
      --job-name=$SLURM_JOB_NAME \
      $PROJECT_HOME/scripts/sbatch_train.sh $BASE_DIR $PROJECT_HOME $PROJECT_BRANCH"
    exit 0
  fi
  echo "Checking if need to cleanup scratch: $NEW_DATA_DIR"
  if [[ -d /scratch ]] && [[ $COPY_DATA -eq 1 ]]; then
    cat "$NEW_DATA_DIR"/data_lock
    # Remove own lock
    sed -i /"$MODEL_NAME"/d "$NEW_DATA_DIR"/data_lock
    if ! wc -l "$NEW_DATA_DIR"/data_lock; then
      # No other locks found. Cleanup space.
      echo "cleaning up data dir $NEW_DATA_DIR"
      rm -r "$NEW_DATA_DIR"
    fi
  fi
}
trap _at_exit EXIT

function _usr1 {
  echo "Caught SIGUSR1 signal!"
  kill -USR1 "$trainprocess" 2>/dev/null
  wait "$trainprocess"
}
trap _usr1 SIGUSR1

cd "$PROJECT_HOME"/DeepFilterNet/df/

# Start training
printf "\n***Starting training***\n\n"

PYTHONPATH="$PROJECT_HOME/DeepFilterNet/" python train.py \
  "$DATA_CFG" \
  "$DATA_DIR" \
  "$BASE_DIR" \
  $DEBUG &

trainprocess=$!
echo "Started trainprocess: $trainprocess"

wait $trainprocess
echo "Training stopped"
