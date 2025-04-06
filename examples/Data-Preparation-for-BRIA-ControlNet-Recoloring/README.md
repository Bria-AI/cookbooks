# Data Preparation for BRIA-ControlNet-Recoloring

A data preparation tool specifically designed for fine-tuning [BRIA-2.3-ControlNet-Recoloring](https://huggingface.co/briaai/BRIA-2.3-ControlNet-Recoloring). This script creates the necessary training pairs: grayscale conditions and color information prompts.

## Overview

The BRIA-2.3-ControlNet-Recoloring model is designed to recolor images based on text prompts. This tool helps prepare training data by:
1. Creating grayscale versions of images (condition images)
2. Extracting dominant colors from original images
3. Formatting color information as training prompts

## Project Structure

```
data/
├── images/      # Original color images
├── condition/   # Grayscale conditions for ControlNet
└── captions/    # Color prompt files
```

## How It Works

1. **Color Extraction**:
   - Uses K-means clustering to identify dominant colors in original images
   - Extracts RGB values that represent the image's color palette
   - Formats colors as RGB triplets for the prompt

2. **Condition Creation**:
   - Converts original images to grayscale
   - These serve as condition images for the ControlNet

3. **Prompt Generation**:
   - Creates text files containing color information
   - Format: "The colors of the image are: RGB(r,g,b), RGB(r,g,b), ..."
   - These prompts can be used for training the model

## Usage

1. Place your training images in the images folder:
```python
data/
└── images/     # Place your original images here
```

2. Run the script:
```bash
python create_data_for_training.py
```

## Generated Data Structure

For each input image (`example.jpg`), the script generates:

1. `condition/example_grayscale.jpg`
   - Grayscale version used as condition input
   - Used by the ControlNet to understand image structure

2. `captions/example_colors.txt`
   - Contains color information in the format:
   ```
   The colors of the image are: RGB(145, 178, 206), RGB(89, 67, 45), RGB(201, 198, 196)
   ```
   - Used as prompts for training

## Integration with BRIA-2.3-ControlNet-Recoloring

The generated data follows the format expected by BRIA-2.3-ControlNet-Recoloring:
- Condition images provide structural information
- Color prompts specify the desired colorization
- Pairs can be used directly for fine-tuning

## Data Format Details

1. **Condition Images**:
   - Grayscale format
   - Preserves original image dimensions (1024*1024)
   - Used as input to the ControlNet

2. **Color Prompts**:
   - Text format with RGB values
   - Extracted from original images
   - Used to guide the recoloring process

## Configuration

Adjust extraction parameters in the script:
```python
# Color extraction settings
n_colors = 5                  # Number of colors to extract
min_cluster_size = 0.005      # Minimum cluster size (as fraction)
```

## Example Training Set

```
data/
├── images/
│   ├── cat.jpg
│   └── dog.jpg              # Original color images
├── condition/
│   ├── cat_grayscale.jpg
│   └── dog_grayscale.jpg    # Conditions for ControlNet
└── captions/
    ├── cat_colors.txt
    └── dog_colors.txt       # Color prompts
```


# Training

The training process is based on the Hugging Face Diffusers ControlNet training script. You'll need to:

1. Download the base training script:
```bash
wget https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet_sdxl.py
```

2. To start training, run:
```bash
# Run the training script
./finetune_controlNet_recoloring.sh
```

Note:
⬆️ The training script is based on [train_controlnet_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet_sdxl.py). 
⬆️ Make sure to fill in the required parameters marked with XX in the script before running.
