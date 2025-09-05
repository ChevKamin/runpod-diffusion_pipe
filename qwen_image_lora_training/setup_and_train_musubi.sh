#!/usr/bin/env bash
set -euo pipefail

########################################
# GPU detection
########################################
gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}'
  elif [ -n "${CUDA_VISIBLE_DEVICES-}" ] && [ "${CUDA_VISIBLE_DEVICES-}" != "" ]; then
    echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}'
  else
    echo 0
  fi
}
GPU_COUNT=$(gpu_count)
echo ">>> Detected GPUs: ${GPU_COUNT}"
if [ "${GPU_COUNT}" -lt 1 ]; then
  echo "ERROR: No CUDA GPUs detected. Aborting."
  exit 1
fi

########################################
# Load user config
########################################
CONFIG_FILE="${CONFIG_FILE:-$(dirname "$0")/musubi_config.sh}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: Config file '$CONFIG_FILE' not found."
  exit 1
fi
source "$CONFIG_FILE"

########################################
# Paths
########################################
WORKDIR="${WORKDIR:-$NETWORK_VOLUME/qwen_image_lora_training}"
DATASET_DIR="${DATASET_DIR:-$WORKDIR/dataset_here}"

REPO_DIR="$WORKDIR/musubi-tuner"
MODELS_DIR="$WORKDIR/models"

SETUP_MARKER="$REPO_DIR/.setup_done"

########################################
# One-time setup
########################################
if [ ! -f "$SETUP_MARKER" ]; then
  echo ">>> Running setup..."

  mkdir -p "$WORKDIR" "$DATASET_DIR" "$MODELS_DIR"/{text_encoders,vae,diffusion_models}

  # Clone Musubi
  cd "$WORKDIR"
  git clone --recursive https://github.com/kohya-ss/musubi-tuner.git "$REPO_DIR" || true
  cd "$REPO_DIR"
  git submodule update --init --recursive

  # Python venv
  apt-get update -y
  apt-get install -y python3-venv
  [ ! -d "venv" ] && python3 -m venv venv
  source venv/bin/activate

  # Python deps
  pip install -e .
  pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118
  pip install huggingface_hub hf_transfer hf_xet
  export HF_HUB_ENABLE_HF_TRANSFER=1

  # Download Qwen-Image models
  echo ">>> Downloading Qwen-Image models..."
  hf download Comfy-Org/Qwen-Image_ComfyUI split_files/diffusion_models/qwen_image_bf16.safetensors \
    --local-dir "$MODELS_DIR/diffusion_models"

  hf download Comfy-Org/Qwen-Image_ComfyUI split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
    --local-dir "$MODELS_DIR/text_encoders"

  hf download Qwen/Qwen-Image vae/diffusion_pytorch_model.safetensors \
    --local-dir "$MODELS_DIR/vae"

  touch "$SETUP_MARKER"
else
  echo ">>> Setup already done. Activating venv."
  cd "$REPO_DIR"
  source venv/bin/activate
fi

########################################
# Dataset config
########################################
mkdir -p "$REPO_DIR/dataset"
DATASET_TOML="$REPO_DIR/dataset/dataset.toml"

cat > "$DATASET_TOML" <<TOML
[general]
resolution = [${RESOLUTION_LIST}]
caption_extension = ".txt"
batch_size = ${BATCH_SIZE}
enable_bucket = true
bucket_no_upscale = false
num_repeats = ${NUM_REPEATS}

[[datasets]]
image_directory = "$DATASET_DIR"
cache_directory = "$DATASET_DIR/cache"
num_repeats = ${NUM_REPEATS}
TOML

echo ">>> dataset.toml written:"
cat "$DATASET_TOML"

########################################
# Pre-caching
########################################
python src/musubi_tuner/qwen_image_cache_latents.py \
  --dataset_config "$DATASET_TOML" \
  --vae "$VAE_PATH"

python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
  --dataset_config "$DATASET_TOML" \
  --text_encoder "$TEXT_ENCODER_PATH" \
  --batch_size 1

########################################
# Training
########################################
mkdir -p "$WORKDIR/output"

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/qwen_image_train_network.py \
    --dit "$DIT_PATH" \
    --vae "$VAE_PATH" \
    --text_encoder "$TEXT_ENCODER_PATH" \
    --dataset_config "$DATASET_TOML" \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift --discrete_flow_shift 2.2 \
    --optimizer_type adamw8bit --learning_rate "$LEARNING_RATE" --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_qwen_image \
    --network_dim "$LORA_RANK" \
    --max_train_epochs "$MAX_EPOCHS" --save_every_n_epochs "$SAVE_EVERY" --seed "$SEED" \
    --output_dir "$WORKDIR/output" --output_name "$TITLE"

echo ">>> Training complete."
