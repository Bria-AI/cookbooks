# Fine-Tune-Bria-4B-Adapt

## Overview
This repository provides step-by-step instructions on how to fine-tune **Bria-4B-Adapt**, a powerful visual generative model. The guide covers downloading necessary files, preparing data, and running the training process efficiently.

## Installation & Setup

### 1. Download Required Files
#### Model Files (from Hugging Face)
Download the required scripts from the [Bria-4B-Adapt model card](https://huggingface.co/briaai/BRIA-4B-Adapt) using the following Python commands:

```python
from huggingface_hub import hf_hub_download
import os

try:
    local_dir = os.path.dirname(__file__)
except:
    local_dir = '.'
    
hf_hub_download(repo_id="briaai/BRIA-4B-Adapt", filename='pipeline_bria.py', local_dir=local_dir)
hf_hub_download(repo_id="briaai/BRIA-4B-Adapt", filename='transformer_bria.py', local_dir=local_dir)
hf_hub_download(repo_id="briaai/BRIA-4B-Adapt", filename='bria_utils.py', local_dir=local_dir)
```

#### Fine-Tuning Code (from this repo)
Download the following fine-tuning scripts and place them in the same directory:
- `run_finetune.py`
- `finetune.py`
- `finetune_configs.py`


### 2. Install Dependencies
Install required packages using:
```bash
pip install -qr https://huggingface.co/briaai/BRIA-4B-Adapt/resolve/main/requirements.txt
```

## Data Preparation
### 1. Data Collection
- Collect at least **a few thousand** image-caption pairs.
- Larger datasets improve model performance but require more training steps.

### 2. Data Format

Store data in a folder containing images and a metadata.csv file with 
captions. This CSV file should have the following columns: 
-  file_name: The file name of each image. 
- caption: The corresponding caption.

The `metadata.csv` file should have:
| file_name | caption |
|-----------|---------|
| image1.jpg | caption for image1 |
| image2.jpg |  caption for image2  |


Ensure your dataset follows this structure:
```
/data_folder/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
├── metadata.csv
```
### 3. Image Considerations
-  **image size:** Train on images around or larger than 1024x1024 pixels. Bria-4B-Adapt 
was trained on multiple aspect ratios around this resolution. 
- The training code performs center-crop and resizes images to 1024x1024 by default. Be 
aware that images with different aspect ratios may lose information at the edges. 
-   Ensure that the images cover the visual domain you intend to train on, maintaining a 
general balance between different elements within this domain. A varied dataset 
promotes better generalization, but ensure consistency in style and content across the 
images.


### 4. Captioning
- If images lack captions, use a **Vision-Language Model (VLM)** to generate them.
- Structure: Include categorizations in the captions for recurring concepts, such as names 
of characters, objects, etc., based on existing classifications within your content. 
○  Implement this by using a specific prefix for each category and asking the VLM to 
complete the description following this prefix
- This approach ensures that during inference, you can trigger the model to 
generate those same recurring concepts using the same prefix. 


## Training
### 1. System Requirements
- Training on 1024x1024 images with `batch_size=3` requires **~80GB VRAM**.
- The recommended setup is **A100 GPUs (8 x 80GB)** for an effective batch size of 24.

### 2. Configure Training Variables
Modify `run_finetune.py` with the following parameters:


```python
 # Path to the data folder (containing images and metadata.csv).
DATA_PATH = "/path/to/data_folder"

# Path to save the trained weights
CHECKPOINT_LOCAL_PATH = "./checkpoints"

 # Default is 3. For 8-GPU machines, the effective batch size is 24
TRAIN_BATCH_SIZE = 3  # Adjust based on available VRAM

 # Default is 1. Increase this value to raise the  effective batch size without needing more VRAM, though this will slow down  training.
GRADIENT_ACCUMULATION_STEPS = 1  # Increase to simulate a larger batch size

MAX_TRAIN_STEPS = 3000  # Increase for larger datasets # Default is 3000. Increase for larger datasets (more than a  couple of thousand images)
CHECKPOINTING_STEPS = 500  # Save model checkpoints every 500 steps
WANDB_PROJECT = "Bria-4B-Finetune"  # (Optional) Weights & Biases tracking
```



### 3. Run Training
Execute the following command to start training:
```bash
torchrun --nproc_per_node=8 examples/bria4B_adapt/run_finetune.py
```

## Contact
For inquiries, please reach out via GitHub issues or the official BriaAI community.
