# ControlNet-ColorGrid
inference for 
https://huggingface.co/briaai/BRIA-2.3-ControlNet-ColorGrid


show color-grid_bria 2.3.png



### Model Card: Fine-tuning BRIA-2.3-ControlNet-ColorGrid

#### Model Summary
This repository contains scripts and instructions for fine-tuning the BRIA-2.3-ControlNet-ColorGrid model, designed to generate art images with fine-grained control over color grids. This model is part of Bria's generative AI solutions, intended to create artistic, visually complex images from structured input prompts and color conditioning.

#### Getting Started with Inference
To use the model, begin with an inference step:

1. **Install Requirements and Set Up Inference:**
   - Access the BRIA models from [here](https://huggingface.co/briaai). Once approved, you can download the model from Hugging Face or proceed with the following instructions.

2. **Log into Hugging Face and Set Up Access Token:**
   - Create a read-only token on Hugging Face:
     - Go to **Settings > Access Tokens**.
     - Select the **Read** type and ensure permissions include **public gated repositories**.
   - Log in using the command below:
     ```bash
     huggingface-cli login
     ```
   - Paste the token when prompted.

3. **Run Inference Script:**
   ```python
   from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
   import torch
   from PIL import Image
   import os

   # Load model and pipeline
   controlnet = ControlNetModel.from_pretrained("briaai/BRIA-2.3-ControlNet-ColorGrid", torch_dtype=torch.float16)
   pipe = StableDiffusionXLControlNetPipeline.from_pretrained("briaai/BRIA-2.3", controlnet=controlnet, torch_dtype=torch.float16)
   pipe.to("cuda")

   # Define prompt and negative prompt
   prompt = "A portrait of a beautiful and playful ethereal singer, golden designs, highly detailed, blurry background"
   negative_prompt = "Logo,Watermark,Text,Ugly,Extra fingers,Mutation,Blurry,Extra limbs"

   # Load and resize input image
   input_image = Image.open('path/to/your/image.png')
   control_image = input_image.resize((16, 16)).resize((1024, 1024), Image.NEAREST)

   # Generate image
   image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=control_image).images[0]

   # Save result
   output_dir = "results"
   os.makedirs(output_dir, exist_ok=True)
   output_path = os.path.join(output_dir, "generated_singer_portrait.png")
   image.save(output_path)
   print(f"Image saved at: {output_path}")
   ```

#### Data Preparation for Training
1. **Data Creation:**
   - Generate captions for art images to describe them for training purposes.
   - Use the BRIA-2.3-FAST model to create initial data, ensuring a variety of art styles and concepts.

2. **Pixelated Image Generation:**
   - Prepare pixelated versions of the images to use as control data, resizing images to 16x16 and back to 1024x1024 using nearest neighbor interpolation.

3. **Data Filtering and Benchmark Design:**
   - Manually filter generated images for quality.
   - Design a benchmark to evaluate the model's performance and ensure consistent output.

#### Uploading Dataset to Hugging Face
For uploading datasets, refer to Hugging Face documentation for step-by-step guidance on creating and uploading datasets for model training.

#### Start Training
- Use the provided script in [this example](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py) to initiate model training.
- Ensure to configure the pipeline with Bria’s preferred foundation models.

