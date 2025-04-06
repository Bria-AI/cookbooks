import os
from typing import List, Optional, Any, Dict

from pydantic import BaseModel, Field
from huggingface_hub import repo_exists


class HyperParemeter(BaseModel):
    pretrained_vae_model_name_or_path: str = Field(
        default="black-forest-labs/FLUX.1-schnell",
        description="Path to an improved VAE to stabilize training. For more details, see https://github.com/huggingface/diffusers/pull/4038.",
    )
    pretrained_text_encoder_name_or_path: str = Field(
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        description="Using it's text encoder google/t5-v1_1-xxl.",
    )
    revision: Optional[str] = Field(
        default=None,
        description="Revision of pretrained model identifier from huggingface.co/models.",
    )
    # TODO: checkpoint saving
    s3_bucket_name: str = Field(
        default="your-s3-bucket",
        description="S3 bucket for saving checkpoints.",
    )
    s3_prefix: str = Field(
        default="your-s3-prefix",
        description="S3 directory saving checkpoints.",
    )

    max_train_samples: Optional[int] = Field(
        default=None,
        description="For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
    )
    seed: int = Field(
        default=10,
        description="A seed for reproducible training.",
    )
    save_metrics: bool = Field(
        default=True,
        description="Save metrics during training.",
    )
    resolution: int = Field(
        default=256,
        description="The resolution for input images.",
    )
    center_crop: int = Field(
        default=0,
        description="Whether to center crop the input images to the resolution.",
    )
    h_flip: int = Field(
        default=0,
        description="Whether to horizontally flip the input images randomly.",
    )
    resize: int = Field(
        default=0,
        description="Resize input images to a specific resolution.",
    )
    max_sequence_length: int = Field(
        default=128,
        description="Maximum sequence length for the T5 text encoder.",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    learning_rate: float = Field(
        default=1e-4,
        description="Initial learning rate to use.",
    )
    lr_scheduler: str = Field(
        default="constant_with_warmup",
        description="The scheduler type to use.",
    )
    lr_warmup_steps: int = Field(
        default=10000,
        description="Number of steps for the warmup in the LR scheduler.",
    )
    allow_tf32: bool = Field(
        default=True,
        description="Whether or not to allow TF32 on Ampere GPUs.",
    )
    weighting_scheme: str = Field(
        default="logit_normal",
        description="Weighting scheme for training.",
    )
    logit_mean: float = Field(
        default=0.0,
        description="Mean to use when using the 'logit_normal' weighting scheme.",
    )
    logit_std: float = Field(
        default=1.0,
        description="Std to use when using the 'logit_normal' weighting scheme.",
    )
    mode_scale: float = Field(
        default=1.29,
        description="Scale of mode weighting scheme.",
    )
    # TODO: this is optimizer config?
    ############################
    use_8bit_adam: bool = Field(
        default=False,
        description="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    use_adafactor: int = Field(
        default=0,
        description="Use AdaFactor optimizer.",
    )
    adam_beta1: float = Field(
        default=0.9,
        description="The beta1 parameter for the Adam optimizer.",
    )
    adam_beta2: float = Field(
        default=0.999,
        description="The beta2 parameter for the Adam optimizer.",
    )
    adam_weight_decay: float = Field(
        default=1e-4,
        description="Weight decay to use.",
    )
    adam_epsilon: float = Field(
        default=1e-08,
        description="Epsilon value for the Adam optimizer.",
    )
    max_grad_norm: float = Field(
        default=1.0,
        description="Max gradient norm.",
    )
    mixed_precision: str = Field(
        default="bf16",
        description="Whether to use mixed precision.",
    )
    noise_offset: float = Field(
        default=0.0,
        description="The scale of noise offset.",
    )

    local_rank: int = Field(
        default=-1,
        description="For distributed training: local_rank.",
    )

    save_pipeline: bool = Field(
        default=False,
        description="Whether to save only the pipeline instead of the entire accelerator.",
    )
    no_cfg: bool = Field(
        default=False,
        description="Avoid replacing 10% of captions with null embeddings.",
    )
    drop_rate_cfg: float = Field(
        default=0.1,
        description="Rate for Classifier Free Guidance dropping.",
    )
    dense_caption_ratio: float = Field(
        default=0.5,
        description="Rate for dense captions.",
    )

    enable_xformers_memory_efficient_attention: bool = Field(
        default=False,
        description="Whether or not to use xformers.",
    )

    first_ema_step: bool = Field(
        default=False,
        description="Initialize EMA model according to the Unet.",
    )
    crops_coords_top_left_h: int = Field(
        default=0,
        description="Coordinate for height to be included in the crop coordinate embeddings needed by SDXL Unet.",
    )
    crops_coords_top_left_w: int = Field(
        default=0,
        description="Coordinate for width to be included in the crop coordinate embeddings needed by SDXL Unet.",
    )
    convert_unet_to_weight_dtype: bool = Field(
        default=False,
        description="Convert Unet to weight_dtype.",
    )
    precompute: bool = Field(
        default=False,
        description="Use precomputed latents and text embeddings.",
    )
    random_latents: int = Field(
        default=0,
        description="Use precomputed latents and text embeddings.",
    )
    reinit_scheduler: int = Field(
        default=0,
        description="Reinitialize the scheduler.",
    )
    reinit_optimizer: int = Field(
        default=0,
        description="Reinitialize the optimizer.",
    )
    reinit_optimizer_type: str = Field(
        default="",
        description="Type of optimizer to reinitialize.",
    )

    train_with_ratios: int = Field(
        default=1,
        description="Train using ratios.",
    )
    # TODO: remove this?
    curated_training: Optional[str] = Field(
        default=None,
        description="Use curated training mode for bucketing.",
    )
    debug: int = Field(
        default=1,
        description="Debug mode flag.",
    )
    gradient_checkpointing: int = Field(
        default=0,
        description="Enable gradient checkpointing.",
    )
    force_download: bool = Field(
        default=True,
        description="Force download from hub.",
    )
    low_res_fine_tune: int = Field(
        default=0,
        description="Low resolution fine-tuning.",
    )
    shift: float = Field(
        default=1.0,
        description="Noise shifting.",
    )
    variant: Optional[str] = Field(
        default=None,
        description="Variant of model files for the pretrained model.",
    )
    use_flow_matching: int = Field(
        default=1,
        description="Use flow matching.",
    )
    compile: int = Field(
        default=0,
        description="Compile transformer.",
    )
    flow_matching_latent_loss: int = Field(
        default=0,
        description="Use latent loss instead of model pred loss.",
    )
    use_continuous_sigmas: int = Field(
        default=0,
        description="Enable continuous sigmas.",
    )
    rope_theta: int = Field(
        default=10000,
        description="Rope frequency.",
    )
    time_theta: int = Field(
        default=10000,
        description="Time embed frequency.",
    )
    use_dynamic_shift: int = Field(
        default=0,
        description="Enable dynamic shift.",
    )

    def get_hyperparameter(self):

        print(f"flow_matching_latent_loss: {self.flow_matching_latent_loss}")

        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank:
            self.local_rank = env_local_rank

        # Sanity checks
        if self.dataset_name is None and self.data_channels == "":
            raise ValueError("Need either a dataset name or a training folder.")

        assert (
            self.s3_prefix is not None
        ), "s3_prefix must be specified, i.e., dir for saving checkpoints at s3"

        # Init boolean args that are ints
        self.reinit_scheduler = self.reinit_scheduler == 1
        self.use_adafactor = self.use_adafactor == 1
        self.reinit_optimizer = self.reinit_optimizer == 1
        self.train_with_ratios == (self.train_with_ratios == 1)

        return self



class DataLoaderConfig(BaseModel):
    """
    Configuration of PyTorch DataLoader.

    For details on the function/meanings of the arguments, refer to:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    batch_size: int = Field(..., description="Number of samples per batch to load.")
    shuffle: bool = Field(
        False, description="Set to True to have the data reshuffled at every epoch."
    )
    sampler: Optional[Any] = Field(
        None, description="Defines the strategy to draw samples from the dataset."
    )
    batch_sampler: Optional[Any] = Field(
        None, description="Sampler to draw batches directly."
    )
    num_workers: int = Field(
        0, description="Number of subprocesses to use for data loading."
    )
    collate_fn: Optional[Any] = Field(
        None, description="Function to merge a list of samples to form a mini-batch."
    )
    pin_memory: bool = Field(
        False,
        description="If True, the data loader will copy Tensors into CUDA pinned memory.",
    )
    drop_last: bool = Field(
        False,
        description="Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.",
    )
    timeout: int = Field(
        0, description="Timeout value in seconds for collecting a batch."
    )
    worker_init_fn: Optional[Any] = Field(
        None,
        description="If not None, this function will be called on each worker subprocess.",
    )
    multiprocessing_context: Optional[Any] = Field(
        None, description="Context for multiprocessing."
    )


class Logger(BaseModel):
    logging_dir: str = Field(
        default="logs",
        description="log directory.",
    )
    report_to: str = Field(default="wandb")


class WandB(Logger):
    wandb_mode: str = Field(
        default="online",
        description="Enable or disable WandB.",
    )
    wandb_project: str = Field(
        default="default",
        description="WandB project name.",
    )
    wandb_entity: str = Field(
        default=None,
        description="WandB entity name.",
    )
    wandb_run_name: str = Field(
        default=None,
        description="WandB run name.",
    )
    wandb_group: str = Field(
        default=None,
        description="WandB group name.",
    )
    wandb_tags: list = Field(
        default=[],
        description="WandB tags.",
    )
    save_images_to_wandb: bool = Field(
        default=False,
        description="Save images and latents to WandB according to save_images_every.",
    )
    report_to: str = Field(
        default="wandb",
        description="The integration to report results and logs.",
    )
    save_images_every: int = Field(
        default=1,
        description="How many steps to save images for WandB.",
    )


class DatasetConfig(BaseModel):
    """
    Configuration of PyTorch Dataset.

    For details on the function/meanings of the arguments, refer to:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """
    local_path: Optional[str] = Field(
        default=None,
        description="Path to a local directory containing a dataset.",
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description=(
            "The name of the Dataset (from the HuggingFace hub) to train on. It can also be a path pointing to a local "
            "copy of a dataset in your filesystem, or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    dataset_config_name: Optional[str] = Field(
        default=None,
        description="The config of the Dataset, leave as None if there's only one config.",
    )
    data_channels: str = Field(
        default="train_1",
        description="A folder containing the training data dirs separated by commas.",
    )
    image_column: str = Field(
        default="image",
        description="The column of the dataset containing an image.",
    )
    caption_column: str = Field(
        default="text",
        description="The column of the dataset containing a caption or a list of captions.",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Path to a directory where the dataset will be cached.",
    )
    train_batch_size: int = Field(
        default=2,
        description="Number of samples per batch to load.",
    )
    random_flip: bool = Field(
        default=False,
        description="Randomly flip the images horizontally.",
    )
    center_crop: bool = Field(
        default=False,
        description="Center crop the images.",
    )
    resolution: int = Field(
        default=256,
        description="Resolution of the images.",
    )


class StrategyConfig(BaseModel):
    """
    Base configuration for distributed training strategies.
    """

    strategy_name: str = Field(
        ..., description="The name of the training strategy (e.g., 'FSDP', 'DDP')."
    )
    devices: int = Field(1, description="Number of devices to use.")
    mixed_precision: bool = Field(False, description="Enable mixed-precision training.")
    gradient_clipping: Optional[float] = Field(
        None, description="Maximum norm for gradient clipping."
    )
    additional_params: Optional[Dict[str, Any]] = Field(
        None, description="Additional parameters specific to the strategy."
    )
    compile: bool = Field(False, description="Compile the model for faster execution.")


class FSDPConfig(StrategyConfig):
    """
    Fully Sharded Data Parallel (FSDP) Configuration.

    Refer to: https://pytorch.org/docs/stable/fsdp.html
    """

    sharding_strategy: Optional[str] = Field(
        None,
        description="Sharding strategy for FSDP (e.g., 'FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD').",
    )
    offload_params: bool = Field(
        False, description="Enable parameter offloading to CPU."
    )
    cpu_offload: bool = Field(True, description="Offload gradients to CPU.")
    auto_wrap_policy: Optional[str] = Field(
        None, description="Policy for auto-wrapping layers for FSDP."
    )
    sync_module_states: bool = Field(
        True,
        description="Synchronize module states across workers during initialization.",
    )
    backward_prefetch: Optional[str] = Field(
        None,
        description="Backward prefetch mode (e.g., 'BACKWARD_PRE', 'BACKWARD_POST').",
    )
    activation_checkpointing: bool = Field(
        False,
        description="Enable activation checkpointing to save memory during training.",
    )


class DDPConfig(StrategyConfig):
    """
    Distributed Data Parallel (DDP) Configuration.

    Refer to: https://pytorch.org/docs/stable/ddp.html
    """

    find_unused_parameters: bool = Field(
        False, description="Find and handle unused parameters in the model."
    )
    bucket_cap_mb: Optional[int] = Field(
        None, description="Bucket size for DDP communication (in megabytes)."
    )
    gradient_as_bucket_view: bool = Field(
        True, description="Use gradient bucket views to reduce memory usage."
    )
    static_graph: bool = Field(
        False, description="Enable static graph optimization for DDP."
    )


class DPConfig(StrategyConfig):
    """
    Data Parallel (DP) Configuration.
    """

    batch_split: Optional[int] = Field(
        None, description="Split batch size across devices for DP."
    )


class SingleDeviceConfig(StrategyConfig):
    """
    Single-Device Training Configuration.
    """

    device_id: Optional[int] = Field(
        None, description="Device ID to use for single-device training."
    )



class TrainerConfig(BaseModel):
    """
    Configuration for the Trainer.
    """

    max_train_steps: int = Field(
        default=250000,
        description="Total number of training steps to perform.",
    )
    resume_from_checkpoint: str = Field(
        default="no",
        description="Whether training should be resumed from a previous checkpoint.",
    )
    log_dir: str = Field("logs/", description="Directory to save logs.")
    checkpoint_every_n_steps: int = Field(
        1000, description="Save a checkpoint every n steps."
    )
    checkpoint_local_path: str = Field(
        default="/tmp/checkpoints",
        description="The local path to save checkpoints to. Then SageMaker will upload from there to S3.",
    )
    base_model_dir: str = Field(
        default="/tmp/models",
        description="The local of the model.",
    )
    checkpointing_steps: int = Field(
        default=5000,
        description="Save a checkpoint of the training state every X updates.",
    )
    num_train_epochs: int = Field(
        default=100,
        description="Number of training epochs.",
    )
    output_dir: str = Field(
        default="/tmp/output",
        description="A path to a directory for storing data.",
    )
    checkpoints_total_limit: Optional[int] = Field(
        default=None,
        description="Maximum number of checkpoints to store.",
    )
    upcast_before_saving: Optional[bool] = Field(
        default=False,
        description="Upcast the model before saving.",
    )
    huggingface_path: Optional[str] = Field(
        default=None,
        description="Path to a directory containing a Hugging Face model.",
    )

    def _download_checkpoint(self, resume_from_checkpoint: str) -> None:
        """
        Download a checkpoint from Hugging Face Hub.
        """
        if self.resume_from_checkpoint == "no" or self.resume_from_checkpoint is None:
            self.checkpoint_dir = "no"

        if not repo_exists(resume_from_checkpoint) and os.access(
            os.path.dirname(resume_from_checkpoint), os.W_OK
        ):
            self.checkpoint_dir = resume_from_checkpoint
            return
        if repo_exists(resume_from_checkpoint):
            self.from_huggingface_hub = True
            return
