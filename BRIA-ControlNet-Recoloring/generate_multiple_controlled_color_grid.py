

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
import torch
from PIL import Image
import os

def create_combined_image(original_image, control_image, generated_images):
    """
    Creates a horizontal concatenation of original image, control image, and multiple generated images.
    Args:
        original_image: The source image
        control_image: The ControlNet input image
        generated_images: List of generated result images
    """
    # Ensure all images are the same height
    height = 1024
    width = 1024
    
    # Number of total images (original + control + generated)
    total_images = 2 + len(generated_images)
    
    # Resize images if needed
    original_image = original_image.resize((width, height))
    control_image = control_image.resize((width, height))
    generated_images = [img.resize((width, height)) for img in generated_images]
    
    # Create a new blank image with space for all images and labels
    total_width = width * total_images
    combined_image = Image.new('RGB', (total_width, height + 30), (255, 255, 255))
    
    # Paste the original and control images
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(control_image, (width, 0))
    
    # Paste the generated images
    for i, gen_image in enumerate(generated_images):
        combined_image.paste(gen_image, ((i + 2) * width, 0))
    
    # Add text labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined_image)
    # Use default font if custom font is not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add labels
    draw.text((width//3, height + 5), "Original Image", fill="black", font=font)
    draw.text((width + width//3, height + 5), "Control Grid", fill="black", font=font)
    for i in range(len(generated_images)):
        draw.text(((i + 2) * width + width//3, height + 5), f"Result {i+1}", fill="black", font=font)
    
    return combined_image

# Load the ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "briaai/BRIA-2.3-ControlNet-ColorGrid",
    torch_dtype=torch.float16
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

# Generate multiple images using different seeds
num_images = 3
generated_images = []
seeds = [42, 123, 567]  # Different seeds for variation

for seed in seeds:
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    # Generate the image
    generated_image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        image=control_image, 
        controlnet_conditioning_scale=1.0,
        height=1024, 
        width=1024,
        num_inference_steps=30,
        generator=torch.Generator(device="cuda").manual_seed(seed)
    ).images[0]
    
    generated_images.append(generated_image)

# Create output directory if it doesn't exist
output_dir = "/home/ubuntu/spring/misc/efrat/color_grid_control_net/res"
os.makedirs(output_dir, exist_ok=True)

# Save individual generated images
for i, img in enumerate(generated_images):
    output_path = os.path.join(output_dir, f"generated_singer_portrait_{i+1}.png")
    img.save(output_path)
    print(f"Generated image {i+1} saved at: {output_path}")

# Create and save the combined image
combined_image = create_combined_image(input_image, control_image, generated_images)
combined_output_path = os.path.join(output_dir, "combined_results.png")
combined_image.save(combined_output_path)
print(f"Combined visualization saved at: {combined_output_path}")
