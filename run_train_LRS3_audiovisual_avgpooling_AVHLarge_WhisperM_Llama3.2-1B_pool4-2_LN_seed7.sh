#!/usr/bin/env bash
# Train LRS3_audiovisual_avg-pooling_AVH-Large_Whisper-M_Llama3.2-1B_pool-4-2_LN_seed7
#
# Recipe decoded from Dr-SHAP-AV's README, which references this exact
# checkpoint identifier for eval_LlamaAVSR.py:
#   - modality:         audiovisual
#   - compression:      avg-pooling
#   - video encoder:    AV-HuBERT Large (frozen -- no lora_avhubert)
#   - audio encoder:    Whisper Medium (openai/whisper-medium.en)
#   - LLM:              Llama-3.2-1B, LoRA-tuned (reduction_lora=32, alpha=4,
#                        matching Dr-SHAP-AV's "--rank 32 --alpha 4" under
#                        this repo's renamed flag -- same hidden_size/rank
#                        bottleneck formula, see models/Llama_LoRA.py)
#   - downsample ratio: audio=4, video=2  ("pool-4-2")
#   - layernorm-projector: True  ("LN")
#   - seed: 7

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EXP_NAME="LRS3_audiovisual_avg-pooling_AVH-Large_Whisper-M_Llama3.2-1B_pool-4-2_LN_seed7_visual_noise"

ROOT_DIR=/ucappell/datasets
EXP_DIR=/aa4825/repos/Llama-AVSR/experiments
PROJECT_WANDB=llama-avsr-training
AVHUBERT_CKPT=/aa4825/models/av_hubert/large_vox_iter5.pt

# LRS3 label files (README options): 30h -> lrs3_30h_train_transcript_lengths_seg16s_LLM_lowercase_12.csv
#                                     433h -> lrs3_train_transcript_lengths_seg16s_LLM_lowercase_25.csv
#                                     1756h (+VoxCeleb2) -> lrs3vox2en_train_transcript_lengths_seg24s_LLM_lowercase_25.csv
# Defaulting to the 433h (LRS3-only) tier; override if you want 30h or +VoxCeleb2.
TRAIN_FILE=lrs3_train_transcript_lengths_seg24s_LLM_lowercase_greater25.csv
VAL_FILE=lrs3_test_transcript_lengths_seg24s_LLM_lowercase.csv
TEST_FILE=lrs3_test_transcript_lengths_seg24s_LLM_lowercase.csv

NUM_NODES=1
GPUS=5
MAX_EPOCHS=10
LR=1e-3

CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_P2P_DISABLE=1 python train.py \
  --exp-dir "$EXP_DIR" \
  --exp-name "$EXP_NAME" \
  --project-wandb "$PROJECT_WANDB" \
  --root-dir "$ROOT_DIR" \
  --train-file "$TRAIN_FILE" \
  --val-file "$VAL_FILE" \
  --test-file "$TEST_FILE" \
  --seed 7 \
  --modality audiovisual \
  --compression-mode avg-pooling \
  --audio-encoder-name openai/whisper-medium.en \
  --pretrain-avhubert-enc-video-path "$AVHUBERT_CKPT" \
  --llm-model meta-llama/Llama-3.2-1B \
  --downsample-ratio-audio 4 \
  --downsample-ratio-video 2 \
  --layernorm-projector True \
  --add_PETF_LLM lora \
  --unfrozen_modules peft_llm \
  --reduction_lora 32 \
  --alpha 4 \
  --num-nodes $NUM_NODES \
  --gpus $GPUS \
  --max-epochs $MAX_EPOCHS \
  --lr $LR \
  --max-frames-audiovisual 1000 \
  --auto-test True
