import os
# @Kobi - remove this path
os.environ["HF_HOME"] = "/vol/biomedic3/bglocker/ugproj2324/nns20/svd-unisim/.cache"
import torch
from diffusers import UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection
from xtend import EmbeddingProjection
from safetensors.torch import load_file
from diffusers.utils import load_image
import numpy as np
from PIL import Image
from pipelines.pipeline_stable_video_diffusion_text_added import StableVideoDiffusionPipeline

# Set environment variable for the cache directory.

def set_device():
    """Check and return the available device (GPU if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(pretrained_model_path, device):
    """
    Load models and components needed for the pipeline.
    
    Args:
    - pretrained_model_path (str): Path to the pretrained models.
    - device (torch.device): Device to load the models onto.
    
    Returns:
    - dict: Dictionary containing the loaded models and components.
    """
    # Load UNet
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
        # variant="fp16",
    )
    unet.embedding_projection = EmbeddingProjection(in_features=1024, hidden_size=1024)
    
    # Load the diffusion model state from safetensors
    state_dict = load_file(os.path.join(pretrained_model_path, 'unet', 'diffusion_pytorch_model.safetensors'))
    
    # Extract the weights for embedding_projection
    embedding_projection_state_dict = {
        'linear_1.weight': state_dict['embedding_projection.linear_1.weight'],
        'linear_1.bias': state_dict['embedding_projection.linear_1.bias']
    }
    
    # Load the weights into the embedding_projection
    unet.embedding_projection.load_state_dict(embedding_projection_state_dict)
    unet.to(device)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder", revision=None, variant="fp16"
    ).to(device)
    
    text_tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="text_encoder").to(device)
    
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", revision=None, variant="fp16"
    ).to(device)

    return {
        'unet': unet,
        'image_encoder': image_encoder,
        'text_tokenizer': text_tokenizer,
        'text_encoder': text_encoder,
        'vae': vae
    }

def load_optimizer_scheduler_states(pretrained_model_path):
    """
    Load the optimizer and scheduler states.
    
    Args:
    - pretrained_model_path (str): Path to the pretrained models.
    
    Returns:
    - tuple: Optimizer state, scheduler state, and random states.
    """
    optimizer_state = torch.load(os.path.join(pretrained_model_path, 'optimizer.bin'))
    scheduler_state = torch.load(os.path.join(pretrained_model_path, 'scheduler.bin'))
    
    # Load random states
    with open(os.path.join(pretrained_model_path, 'random_states_0.pkl'), 'rb') as f:
        random_states = torch.load(f)
    
    return optimizer_state, scheduler_state, random_states

def create_pipeline(models, device):
    """
    Create and return the stable video diffusion pipeline.
    
    Args:
    - models (dict): Dictionary containing the loaded models and components.
    - device (torch.device): Device to load the pipeline onto.
    
    Returns:
    - StableVideoDiffusionPipeline: The initialized video diffusion pipeline.
    """
    return StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        unet=models['unet'],
        image_encoder=models['image_encoder'],
        vae=models['vae'],
        text_encoder=models['text_encoder'],
        tokenizer=models['text_tokenizer'],
        revision=None,
        torch_dtype=torch.float32,
    ).to(device)

def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.
    
    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.
    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=1000 / fps,  # Duration in milliseconds
                       loop=0)

def run_pipeline(pipeline, prompts, num_frames, input_image_path):
    """
    Run the video diffusion pipeline for given prompts and save output as GIF.
    
    Args:
    - pipeline (StableVideoDiffusionPipeline): The video diffusion pipeline.
    - prompts (list): List of text prompts for the pipeline.
    - num_frames (int): Number of frames in the output video.
    - input_image_path (str): Path to the input image.
    """
    for prompt in prompts:
        video_frames = pipeline(
            load_image(input_image_path).resize((256, 256)),
            prompt,
            height=256,
            width=256,
            num_frames=num_frames,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=7,
            noise_aug_strength=0.02
        ).frames[0]

        out_file = f"evaluation/outputs/{prompt}.gif"
        
        for i in range(num_frames):
            img = video_frames[i]
            video_frames[i] = np.array(img)
        
        export_to_gif(video_frames, out_file, 7)

def main():
    args = {
        # @Kobi - change your path to the pretrained model here
        'pretrained_model_name_or_path': '/vol/biomedic3/bglocker/ugproj2324/nns20/svd-unisim/saved_checkpoints/checkpoint-205000-big-dataset',
        'revision': None,
    }

    device = set_device()
    models = load_models(args['pretrained_model_name_or_path'], device)
    # what are these optimizer, scheduler and random states meant to do
    optimizer_state, scheduler_state, random_states = load_optimizer_scheduler_states(args['pretrained_model_name_or_path'])
    pipeline = create_pipeline(models, device)
    
    image_prompt_file = "evaluation/in_distribution_actions_extended.txt"
    with open(image_prompt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            image = line.split(",")[0]
            prompt = line.split(",")[1]
            run_pipeline(pipeline, [prompt], num_frames=25, input_image_path=image)

if __name__ == "__main__":
    main()