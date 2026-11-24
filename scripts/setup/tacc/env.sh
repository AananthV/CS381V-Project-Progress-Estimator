# Source on TACC before any project command (setup, jobs): tools in $WORK, caches off $HOME.
# Set TACC_TOOLS and HF_HOME first (e.g. under $SCRATCH) when $WORK is near its quota; sbatch passes them on.
TACC_TOOLS=${TACC_TOOLS:-$WORK/tools}
export PATH="$TACC_TOOLS/uv:$PATH"
export UV_PYTHON_INSTALL_DIR="$TACC_TOOLS/python"
export UV_CACHE_DIR="$SCRATCH/cache/uv"
export UV_LINK_MODE=copy
export UV_PYTHON_PREFERENCE=only-managed  # the system python3.12 has no headers (evdev cannot build)
export HF_HOME="${HF_HOME:-$WORK/cache/huggingface}"
export XDG_CACHE_HOME="$SCRATCH/cache"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Login nodes cap virtual memory (8 GB) and processes (300): keep uv's thread pools small.
export RAYON_NUM_THREADS=4 UV_CONCURRENT_INSTALLS=2 UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=1
