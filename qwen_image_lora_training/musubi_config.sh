# ====== Qwen-Image LoRA Config File ======

# LoRA hyperparameters
LORA_RANK=16
MAX_EPOCHS=16
SAVE_EVERY=1
LEARNING_RATE=5e-5
SEED=42

# Dataset
DATASET_TYPE=image
RESOLUTION_LIST="1024, 1024"
DATASET_DIR="$NETWORK_VOLUME/qwen_image_lora_training"

# Output
TITLE="Qwen_Image_LoRA"

# Training batch settings
BATCH_SIZE=1
NUM_REPEATS=1

# Qwen model paths
VAE_PATH="$NETWORK_VOLUME/models/vae/diffusion_pytorch_model.safetensors"
TEXT_ENCODER_PATH="$NETWORK_VOLUME/models/text_encoders/qwen_2.5_vl_7b.safetensors"
DIT_PATH="$NETWORK_VOLUME/models/diffusion_models/qwen_image_bf16.safetensors"
