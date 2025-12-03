import torch
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16*.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE*.pth", offload_device="cpu"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/Wan2.2-TI2V-5B_lora_fg1/epoch-4.safetensors", alpha=1)
pipe.enable_vram_management()
input_image = VideoData("data/data_fg/videos/forrest1.mp4", height=480, width=832)[0]

video = pipe(
    prompt="Forrest running",
    negative_prompt="",
    input_image=input_image,
    num_frames=49,
    seed=1, tiled=True,
)
save_video(video, "video_Wan2.2-TI2V-5B.mp4", fps=15, quality=5)
