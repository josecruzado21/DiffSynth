#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --partition=willett-gpu
#SBATCH --gpus=1
#SBATCH --nodelist=g5
#SBATCH --cpus-per-task=1

eval "$(/share/data/willett-group/jcruzado/miniconda/bin/conda shell.bash hook)"
conda activate wan

export HOME=/share/data/speech-lang/users/jcruzado/fake_home
mkdir -p $HOME
export MODELSCOPE_CACHE=$HOME/.cache/modelscope
export MODEL_SCOPE_HOME=$HOME/.cache/modelscope
export MODELSCOPE_HOME=$HOME/.cache/modelscope
export MODELSCOPE_GLOBAL_CACHE=$HOME/.cache/modelscope
export MODELSCOPE_CONFIG_DIR=$HOME/.cache/modelscope
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export XDG_CACHE_HOME=$HOME/.cache
mkdir -p $MODELSCOPE_CACHE
mkdir -p $HF_HOME
export MODELSCOPE_CACHE=/share/data/speech-lang/users/jcruzado/.cache/modelscope
export MODELSCOPE_MODEL_DIR=/share/data/speech-lang/users/jcruzado/repos/Wan2.2/Wan2.2-TI2V-5B


cd /share/data/speech-lang/users/jcruzado/repos/DiffSynth-Studio

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/data_fg \
  --dataset_metadata_path data/data_fg/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16*.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE*.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_lora_fg1" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image"
