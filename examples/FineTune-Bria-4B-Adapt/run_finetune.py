import datetime
import sys
import os

from finetune_configs import HyperParemeter as model_config
from finetune_configs import DataLoaderConfig
from finetune_configs import WandB
from finetune_configs import DatasetConfig
from finetune_configs import FSDPConfig
from finetune_configs import TrainerConfig
from finetune import Bria4BAdapt as Model

train_batch_size = 3
gradient_accumulation_steps=1
max_train_steps = 3000
checkpointing_steps = 250
wandb_project = "ea-finetune"
data_path = "/mnt/datasets/fox"
checkpoint_local_path = "/mnt/checkpoints"


if __name__ == "__main__":
    #add env variable
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
    os.environ["ACCELERATE_DYNAMO_BACKEND"] = "NO"
    os.environ["ACCELERATE_DYNAMO_MODE"] = "default"
    os.environ["ACCELERATE_DYNAMO_USE_FULLGRAPH"] = "False"
    os.environ["ACCELERATE_DYNAMO_USE_DYNAMIC"] = "False"
    os.environ["ACCELERATE_USE_FSDP"] = "True"
    

    print("Running example_train.py")
    bria_conf = model_config(
        train_batch_size=train_batch_size,
        precompute=False,  
        gradient_accumulation_steps=gradient_accumulation_steps,
        flow_matching_latent_loss=0,
        shift=4.0,
        mixed_precision="bf16",
        # use_flow_matching=1, #?
        # use_dynamic_shift=0,
        # train_with_ratios=0,
        lr_warmup_steps=100,  # ?
        resolution=1024,
        resize=True,
        center_crop=True,
        weighting_scheme="uniform",
        force_download=False,
        max_grad_norm=1.0,
        adam_weight_decay=1e-04,
        adam_epsilon=1e-08,
        learning_rate=5e-05,
        random_latents=0,
        train_with_ratios=0,
        reinit_scheduler=0,
        reinit_optimizer=0,
        pretrained_vae_model_name_or_path="briaai/BRIA-4B-Adapt",
        pretrained_text_encoder_name_or_path="briaai/BRIA-4B-Adapt"
    )

    model = Model(bria_conf)

    dataset = DatasetConfig(
        local_path=data_path,
        caption_column="caption",
        image_column="file_name",
        train_batch_size=train_batch_size,
    )

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    logger = WandB(
        wandb_mode="online",
        wandb_project=wandb_project,
        wandb_run_name=f"test-{time}",
    )

    # Trainer
    trainer = TrainerConfig(
        max_train_steps=max_train_steps,
        # resume_from_checkpoint="/home/ubuntu/gradient/checkpoints/BRIA-4B-Adapt",
        # base_model_dir="/home/ubuntu/BRIA-4B-Adapt/transformer",
        checkpoint_local_path=checkpoint_local_path,
        checkpointing_steps=checkpointing_steps,
        huggingface_path="briaai/BRIA-4B-Adapt",
    )

    # Startegey
    strategy = FSDPConfig(
        strategy_name="fsdp",
        cpu_offload=True,
    )

    # Train
    model.train(
        dataset_config=dataset,
        trainer_config=trainer,
        logger_config=logger,
        startegy_config=strategy,
        dataloader_config=DataLoaderConfig(num_workers=1, batch_size=train_batch_size),
    )
