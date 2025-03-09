'''
Start with installing required package:
pip install diffusers
huggingface-cli login
Get  token from https://huggingface.co/settings/tokens/new?tokenType=read
'''

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
import torch
from PIL import Image
import os

# Load the ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "briaai/BRIA-2.3-ControlNet-ColorGrid",
    torch_dtype=torch.float16  # Using float16 for memory efficiency
)

# Initialize Stable Diffusion XL Pipeline with ControlNet
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "briaai/BRIA-2.3",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
# Move model to GPU
pipe.to("cuda")

# Define the desired prompt for image generation
prompt = "A portrait of a Beautiful and playful ethereal singer, golden designs, highly detailed, blurry background"

# Define negative prompt - what we don't want in the image
negative_prompt = "Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers"

# Load and prepare the input image
input_image = Image.open('/home/ubuntu/spring/misc/efrat/color_grid_control_net/singer.png')
# Resize to 16x16 and then back to 1024x1024 using nearest neighbor interpolation
control_image = input_image.resize((16, 16)).resize((1024,1024), Image.NEAREST)

# Generate the image using the pipeline
image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt, 
    image=control_image, 
    controlnet_conditioning_scale=1.0,  # Control the influence of ControlNet
    height=1024, 
    width=1024
).images[0]

# Create output directory if it doesn't exist
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Save the generated image with a descriptive filename
output_path = os.path.join(output_dir, "generated_singer_portrait.png")
image.save(output_path)
print(f"Image saved successfully at: {output_path}")

