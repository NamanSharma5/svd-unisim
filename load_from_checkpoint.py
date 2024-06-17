import os
os.environ["HF_HOME"] = "/vol/biomedic3/bglocker/ugproj2324/nns20/svd-unisim/.cache"
import torch
from diffusers import UNetSpatioTemporalConditionModel,AutoencoderKLTemporalDecoder
from pipelines.pipeline_stable_video_diffusion_text_added import StableVideoDiffusionPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel
from xtend import EmbeddingProjection
from safetensors.torch import load_file
from diffusers.utils import load_image
import numpy as np
from PIL import Image

def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


# Paths and arguments
args = {
    'pretrained_model_name_or_path': '/vol/biomedic3/bglocker/ugproj2324/nns20/svd-unisim/outputs/checkpoint-100',
    # 'pretrained_model_name_or_path': '/vol/biomedic3/bglocker/ugproj2324/nns20/svd-unisim/outputs_2/checkpoint-61000',
    # 'pretrained_model_name_or_path': '/vol/biomedic3/bglocker/ugproj2324/nns20/svd-unisim/outputs_2/checkpoint-19000',
    'revision': None,
    'pretrain_unet': None
}

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load UNet
unet = UNetSpatioTemporalConditionModel.from_pretrained(
    args['pretrained_model_name_or_path'] if args['pretrain_unet'] is None else args['pretrain_unet'],
    subfolder="unet",
    low_cpu_mem_usage=True,
    # variant="fp16",
)

unet.embedding_projection = EmbeddingProjection(
        in_features=1024, hidden_size=1024,
)

unet.to(device)

# Load the diffusion model state from safetensors
state_dict = load_file(os.path.join(args['pretrained_model_name_or_path'], 'unet', 'diffusion_pytorch_model.safetensors'))

# Extract the weights for embedding_projection
embedding_projection_state_dict = {
    'linear_1.weight': state_dict['embedding_projection.linear_1.weight'],
    'linear_1.bias': state_dict['embedding_projection.linear_1.bias']
}

# Load the weights into the embedding_projection
unet.embedding_projection.load_state_dict(embedding_projection_state_dict)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder", revision=args["revision"], variant="fp16"
).to(device)

text_tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1",subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1",subfolder="text_encoder").to(device)

# Load optimizer state
optimizer_state = torch.load(os.path.join(args['pretrained_model_name_or_path'], 'optimizer.bin'))

# Load scheduler state
scheduler_state = torch.load(os.path.join(args['pretrained_model_name_or_path'], 'scheduler.bin'))

vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", revision=args['revision'], variant="fp16").to(device)

# Load random states
with open(os.path.join(args['pretrained_model_name_or_path'], 'random_states_0.pkl'), 'rb') as f:
    random_states = torch.load(f)

weight_dtype = torch.float32
pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    unet=unet,
    image_encoder= image_encoder,
    vae=vae,
    text_encoder=text_encoder,    
    tokenizer = text_tokenizer,                            
    revision=None,
    torch_dtype=weight_dtype,
).to(device)

# test a forward pass
prompts = ["grab cheese", "grab knife"]
prompt = "grab cheese"
num_frames = 25
video_frames = pipeline(
    load_image(f'demo_2.jpg').resize((256, 256)),
    prompt,
    height=256,
    width=256,
    num_frames=num_frames,
    decode_chunk_size=8,
    motion_bucket_id=127,
    fps=7,
    noise_aug_strength=0.02,
).frames[0]

out_file = "test_checkpoint_forward_pass.gif"

for i in range(num_frames):
    img = video_frames[i]
    video_frames[i] = np.array(img)
export_to_gif(video_frames, out_file, 8)



