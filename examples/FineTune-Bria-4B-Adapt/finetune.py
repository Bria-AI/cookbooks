import itertools
from pathlib import Path
from torch.utils.data import Dataset
from accelerate.logging import get_logger
from PIL.ImageOps import exif_transpose
from PIL import Image

import functools
import logging
import math
import os
from datetime import datetime
from typing import Optional
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from datasets import load_dataset
from diffusers import AutoencoderKL  # Waiting for diffusers udpdate
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder
from torchvision import transforms
from tqdm.auto import tqdm

import wandb

# sagemaker_ssh_helper.setup_and_start_ssh()
import json
import random
from datetime import date, timedelta
import glob
from typing import List

import torch.distributed as dist
import webdataset as wds
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)
from transformers import T5EncoderModel, T5TokenizerFast


from finetune_configs import HyperParemeter as model_config
from finetune_configs import DataLoaderConfig, DatasetConfig, TrainerConfig, StrategyConfig, Logger


from bria_utils import get_t5_prompt_embeds
# from bucket_spliting import (
#     get_deterministic_training_dirs,
#     get_deterministic_training_dirs_dynamic_batches,
# )
from pipeline_bria import BriaPipeline
# from gradient.models.diffusion.bria4B_adapt.model_utils.schedulers import BriaDDIMScheduler
from transformer_bria import (
    BriaTransformer2DModel,
)

logger = get_logger(__name__)

class FinetuneDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        dataset_name=None,
        image_column=None,
        caption_column=None,
        class_data_root=None,
        class_num=None,
        resolution=256,
        repeats=1,
        center_crop=False,
        resize=False,
        h_flip=False,
    ):
        self.resolution = resolution
        self.center_crop = center_crop
        self.resize = resize
        self.h_flip = h_flip

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.dataset_name = dataset_name
        self.image_column = image_column
        self.caption_column = caption_column


        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if self.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                self.dataset_name,
                None,
                None,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if self.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = self.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{self.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]

            if self.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if self.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{self.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][self.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir()) if path.is_file()]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        self.pixel_values = []

        transforms_todo = []
        if self.center_crop:
            print('Adding center_crop')
            transforms_todo +=[transforms.Resize(self.resolution,interpolation=transforms.InterpolationMode.BILINEAR)]
            transforms_todo +=[transforms.CenterCrop(self.resolution)]
        elif self.resize:
            print('Adding resize')
            transforms_todo += [transforms.Resize(self.resolution)]
        
        if self.h_flip:
            print('Adding horizontal flip')
            transforms_todo+=[transforms.RandomHorizontalFlip(0.5)]

        transforms_todo+=[
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
        
        train_transforms = transforms.Compose(transforms_todo)

        for image in self.instance_images:
            image = train_transforms(image.convert('RGB'))

            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        example["pixel_values"] = instance_image

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["caption"] = caption
            else:
                example["caption"] = self.instance_prompt

        else:  # custom prompts were provided, but length does not match size of image dataset
            example["caption"] = self.instance_prompt

        return example


def load_dataset_from_tars(
    training_dirs: List[str],
    rank: int,
    world_size: int,
    seed: int = 0,
    slice: bool = False,
):
    tar_files = []
    for train_data_dir in training_dirs:
        files = glob.glob(f"{train_data_dir}/*.tar")
        print(f"We have {len(files)} data files on {train_data_dir}")
        tar_files += files

    total = len(tar_files)

    print(f"We have {total} data files in total")

    if slice:
        print("Using slicing")
        tar_files = tar_files[
            rank::world_size
        ]  # starting from rank skip world size and take object at each step

        print(f"Process {rank} will use data files {len(tar_files)} files")

    np.random.seed(seed)
    np.random.shuffle(tar_files)

def initialize_distributed():
    # Initialize the process group for distributed training
    dist.init_process_group("nccl")

    # Get the current process's rank (ID) and the total number of processes (world size)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Initialized distributed training: Rank {rank}/{world_size}")


class CudaTimerContext:
    def __init__(self, times_arr):
        self.times_arr = times_arr

    def __enter__(self):
        self.before_event = torch.cuda.Event(enable_timing=True)
        self.after_event = torch.cuda.Event(enable_timing=True)
        self.before_event.record()

    def __exit__(self, type, value, traceback):
        self.after_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.before_event.elapsed_time(self.after_event) / 1000
        self.times_arr.append(elapsed_time)


def get_env_prefix():
    env = os.environ.get("CLOUD_PROVIDER", "AWS").upper()
    if env == "AWS":
        return "SM_CHANNEL"
    elif env == "AZURE":
        return "AZUREML_DATAREFERENCE"
    else:
        return ""


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu"
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
    "timm/imagenet-1k-wds": ("jpg", "cls"),
    "imagenet-1k": ("image", "label"),
    "Maysee/tiny-imagenet": ("image", "label"),
}


class Bria4BAdapt:
    def __init__(self, model_config: model_config):
        self.args = model_config

    def train(
        self,
        dataset_config: DatasetConfig,
        trainer_config: TrainerConfig,
        startegy_config: StrategyConfig,
        dataloader_config: DataLoaderConfig,
        logger_config: Optional[Logger] = None,
    ):

        args = self.args
        set_seed(args.seed)
        logger = get_logger(__name__, log_level="INFO")
        if logger_config is None:
            logger_config = Logger()

        logging_dir = os.path.join(
            trainer_config.checkpoint_local_path, logger_config.logging_dir
        )
        accelerator_project_config = ProjectConfiguration(
            project_dir=trainer_config.checkpoint_local_path, logging_dir=logging_dir
        )

        fsdp_plugin = None
        print(f"Strategy name: {startegy_config.strategy_name}")
        if startegy_config.strategy_name == "fsdp":
            os.environ["ACCELERATE_USE_FSDP"] = "true"
            if startegy_config.compile:
                os.environ["FSDP_USE_ORIG_PARAMS"] = "true"

            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(
                    offload_to_cpu=startegy_config.cpu_offload,
                    rank0_only=True,
                ),
                optim_state_dict_config=FullOptimStateDictConfig(
                    offload_to_cpu=True, rank0_only=True
                ),
                # sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, #FULL_SHARD, #SHARD_GRAD_OP,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # FULL_SHARD, #SHARD_GRAD_OP,
                auto_wrap_policy=functools.partial(
                    size_based_auto_wrap_policy,
                    min_num_params=int(0.5e8),  # 2e-08 ~ 4.47 secs
                ),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            )
            kwargs_handlers = None
            kwargs_handlers = [
                InitProcessGroupKwargs(timeout=timedelta(1000))
            ]  # Insetad of default 600
            print("Using fsdp")
        else:
            kwargs_handlers = [
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ]

        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=logger_config.report_to,
            project_config=accelerator_project_config,
            # dispatch_batches=False,
            fsdp_plugin=fsdp_plugin,
            kwargs_handlers=kwargs_handlers,
        )

        # Disable AMP for MPS.
        # DO we need this ?
        if torch.backends.mps.is_available():
            # if args.use_fsdp:
            print("Disabling accelerator.native_amp")
            accelerator.native_amp = False

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        # Set huggingface token key if provided
        with accelerator.main_process_first():
            if accelerator.is_local_main_process:
                if os.environ.get("HF_API_TOKEN"):
                    HfFolder.save_token(os.environ.get("HF_API_TOKEN"))

        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if not args.precompute:
            tokenizer = T5TokenizerFast.from_pretrained(
                trainer_config.huggingface_path,
                subfolder="tokenizer",
                # force_download=args.force_download,
            )
            text_encoder = T5EncoderModel.from_pretrained(
                trainer_config.huggingface_path,
                subfolder="text_encoder",
                torch_dtype=torch.float16,
                # force_download=args.force_download,
            )

            # T5 is senstive to precision so we use the precision used for precompute and cast as needed
            for block in text_encoder.encoder.block:
                block.layer[-1].DenseReluDense.wo.to(dtype=torch.float32)

            vae = AutoencoderKL.from_pretrained(
                trainer_config.huggingface_path,
                subfolder="vae",
                revision=args.revision,
                # force_download=args.force_download,
            )
        else:
            print("using precompted datasets")
            vae = AutoencoderKL.from_pretrained(
                trainer_config.huggingface_path,
                subfolder="vae",
                revision=args.revision,
                device="cpu",
                # force_download=args.force_download,
            )

        dynamic_batches = None

        # Load scheduler and models
        if args.use_flow_matching:
            print("Using FlowMatching")
            assert (
                args.noise_offset == 0
            )  # we have a zero snr schedule so no need for this
            # TODO - Enable/Disbale Dynamic shifting ?

            base_dir = os.path.dirname(os.path.abspath(__file__))
            if args.use_dynamic_shift:
                dynamic_batches = {256 * 256: 32, 512 * 512: 12, 1024 * 1024: 3}

                assert args.train_with_ratios == 1  # Change batch on the fly
                assert args.shift == 3  # Only using default config
                assert args.gradient_accumulation_steps == 1
                print("using orig flux scheduler")
                config_path = os.path.join(
                    base_dir, "model_utils/flux_scheduler_orig.json.out"
                )
                with open(config_path) as f:
                    scheduler_config = json.load(f)

                noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                    scheduler_config
                )
                

            else:
                # config_path = os.path.join(
                #     base_dir, "model_utils/flux_scheduler.json.out"
                # )
                # with open(config_path) as f:
                #     scheduler_config = json.load(f)

                # noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                #     scheduler_config, shift=args.shift, use_dynamic_shifting=False
                # )
                noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(trainer_config.huggingface_path, subfolder='scheduler', shift=4, use_dynamic_shifting=False)

            # Not sure why a copy is needed
            # noise_scheduler_copy = copy.deepcopy(noise_scheduler)
            print(
                f"Scheduler shift - {noise_scheduler.shift}, dynamic - {noise_scheduler.use_dynamic_shifting}, weighting_scheme - {args.weighting_scheme}"
            )
        else:
            # Maybe take this shift to ddpm ?
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(
                base_dir, "model_utils/scheduler_diffusion.json.out"
            )
            with open(config_path) as f:
                scheduler_config = json.load(f)

                # what to put here on model type ???
            if args.shift > 1:
                noise_scheduler = BriaDDIMScheduler.from_config(
                    scheduler_config, shift=args.shift
                )
                print(f"Using diffusion with shift {noise_scheduler.shift}")
            else:
                noise_scheduler = DDIMScheduler.from_config(
                    scheduler_config,
                )
                print("Using diffusion")

        # base_dir = os.path.dirname(os.path.abspath(__file__))
        # if args.debug:
        #     config_path = os.path.join(
        #         base_dir, "model_utils/flux_transformer_debug.json.out"
        #     )
        # else:
        #     config_path = os.path.join(
        #         base_dir, "model_utils/flux_transformer.json.out"
        #     )

        # with open(config_path) as f:
        #     transformer_config = json.load(f)

        # print("------- Transforemr Config---------")
        # print(transformer_config)

        # print(f"Rope theta: {args.rope_theta}, Time max period: {args.time_theta}")
        # transformer = BriaTransformer2DModel.from_config(
        #     transformer_config, time_theta=args.time_theta, rope_theta=args.rope_theta
        # )
        # # transformer_init = os.environ.get(f"{get_env_prefix()}_TRANSFORMER_INIT")
        # if trainer_config.huggingface_path:
        #     print(f"Loading transformer from {trainer_config.huggingface_path}")
        #     transformer.from_pretrained(
        #         trainer_config.huggingface_path,
        #         subfolder="transformer",
        #         force_download=args.force_download,
        #     )
        # else:
        #     transformer_init = trainer_config.base_model_dir
        #     if transformer_init:
        #         if os.path.exists(
        #             f"{transformer_init}/diffusion_pytorch_model.safetensors"
        #         ):

        #             transformer_init = (
        #                 f"{transformer_init}/diffusion_pytorch_model.safetensors"
        #             )
        #         else:
        #             transformer_init = f"{transformer_init}/pytorch_model_fsdp.bin"

        #         print(
        #             f"\n--------Loading transformer weights from {transformer_init}--------\n"
        #         )

        #         transformer.load_state_dict(torch.load(transformer_init))

        transformer = BriaTransformer2DModel.from_pretrained(trainer_config.huggingface_path, 
                                                             subfolder='transformer',
                                                            #   force_download=args.force_download
                                                              )

        transformer.to(accelerator.device)
        if args.compile:
            transformer = torch.compile(transformer)

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        print(f"Using precision of {weight_dtype}")

        if not args.precompute:
            # T5 given different results on fp16 and bf16 so want to use it like in inference
            text_encoder.to(accelerator.device)
            # vae might be less stable on fp16
            assert vae.dtype == torch.float32
            vae.to(accelerator.device)

            vae.requires_grad_(False)
            text_encoder.requires_grad_(False)

            # Need to be fp16 like in inference due to senstivity to precision
            assert text_encoder.dtype == torch.float16

        transformer.requires_grad_(True)
        # Need to be in float 32 for mixed precision training
        assert transformer.dtype == torch.float32

        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()

        if args.enable_xformers_memory_efficient_attention:  # FIXME
            if is_xformers_available():
                transformer.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )
        if startegy_config.strategy_name == "fsdp":
            transformer = accelerator.prepare(transformer)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Initialize the optimizer
        if args.use_8bit_adam:
            print("Using 8bit adam")
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )
            optimizer_cls = bnb.optim.AdamW8bit
        elif args.use_adafactor:
            optimizer_cls = transformers.Adafactor
        else:
            optimizer_cls = torch.optim.AdamW
        if optimizer_cls == transformers.Adafactor:
            print("using adafactor")
            optimizer = optimizer_cls(
                transformer.parameters(), lr=args.learning_rate, relative_step=False
            )
        else:
            optimizer = optimizer_cls(
                transformer.parameters(),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )

        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
        RANK = int(os.environ.get("RANK", 0))

        # Collect data dirs
        training_dirs = []
        training_dirs_suffixes = dataset_config.data_channels.replace(" ", "").split(
            ","
        )

        for train_dir in training_dirs_suffixes:
            if train_dir.lower() == "resume_from_checkpoint":
                continue
            if train_dir.lower() == "null_condition_embedding":
                continue

            if train_dir.lower() == "models":
                continue

            if train_dir.lower() == "transformer_init":
                continue

            train_data_dir = os.environ.get(f"{get_env_prefix()}_{train_dir.upper()}")
            if train_data_dir:
                training_dirs += [train_data_dir]

        # Seed for the data shuffeling
        # Since we resuem runs we want different data on each resume to make sure we go over all the data
        seed = args.seed + RANK
        set_seed(seed)
        random.seed(seed)

        print(f"Hello, Im Rank {RANK}, from world size {WORLD_SIZE}")

        # Must be ratios or datasetname
        if dataset_config.dataset_name and args.train_with_ratios:
            raise Exception("Please choose ratios or datasetname")

        if (
            dataset_config.dataset_name is None
            and dataset_config.local_path is None
            and args.center_crop
        ):
            raise Exception("center_crop is only used with dataset_name")

        if (
            dataset_config.dataset_name is None
            and dataset_config.local_path is None
            and args.resize
        ):
            raise Exception("resize is only used with dataset_name")

        if (
            dataset_config.dataset_name is None
            and dataset_config.local_path is None
            and args.h_flip
        ):
            raise Exception("resize is only used with dataset_name")

        # # Multi Aspect Ratio
        # if args.train_with_ratios:
        #     # All ratio folder must be in the format of ratio_xxx_width_xxx_height_xxx
        #     training_dirs = sorted(training_dirs)

        #     def remove_slash_suffix(p):
        #         if p[-1] == "/":
        #             p = p[:-1]
        #         return p

        #     training_dirs = [remove_slash_suffix(p) for p in training_dirs]

        #     # make sure all training dirs are in proper format
        #     def get_ratio_width_height(p):

        #         p = os.path.basename(p)

        #         # assert p.startswith('ratio_') and 'width' in p and 'height' in p
        #         print(p)
        #         # Azure WorkAround
        #         azure_input_extenstion = "INPUT_"
        #         p = p.replace(azure_input_extenstion, "")
        #         _, ratio, _, width, _, height = p.split("_")
        #         ratio = ratio
        #         width = int(width)
        #         height = int(height)
        #         return ratio, width, height

        #     # Check input is in p[roper format]
        #     try:
        #         for path in training_dirs:
        #             ratio, width, height = get_ratio_width_height(path)
        #             if args.dense_caption_ratio > 0:
        #                 print(f"Using dense ratio of {args.dense_caption_ratio}")
        #                 assert os.path.exists(f"{path}/dense_captions")

        #             if args.dense_caption_ratio < 1:
        #                 assert os.path.exists(f"{path}/captions")

        #     except Exception as e:
        #         print("Training dir for ratio must be in format ratio_xxx_width_height")
        #         raise e

        #     if WORLD_SIZE < len(
        #         training_dirs
        #     ):  # Happens only on testing with 1 instance
        #         print("Using stochestic bucket")
        #         raise Exception("Not implemented")
        #         bucket_index = get_bucket_index_stochastic(
        #             RANK, training_dirs
        #         )  # Just for testing
        #     else:
        #         if args.curated_training:
        #             print("Using curated buckets")
        #             raise Exception("Not implemented")
        #             if args.curated_training == "get_bucket_index_curated":
        #                 bucket_index = get_bucket_index_curated(
        #                     RANK, WORLD_SIZE, training_dirs
        #                 )  # Just for testing
        #             elif args.curated_training == "get_bucket_index_curated_hr":
        #                 bucket_index = get_bucket_index_curated_hr(
        #                     RANK, WORLD_SIZE, training_dirs
        #                 )  # Just for testing
        #             else:
        #                 raise Exception("Curated bucketing not found")
        #         else:
        #             print("Using deterministic buckets")
        #             if dynamic_batches:
        #                 training_dirs = get_deterministic_training_dirs_dynamic_batches(
        #                     RANK,
        #                     WORLD_SIZE,
        #                     training_dirs,
        #                     args.dense_caption_ratio,
        #                     dynamic_batches,
        #                 )
        #             else:
        #                 training_dirs = get_deterministic_training_dirs(
        #                     RANK,
        #                     WORLD_SIZE,
        #                     training_dirs,
        #                     args.dense_caption_ratio,
        #                     dynamic_batches,
        #                 )

        #             training_dirs = [training_dirs]

        #     # Get chosen bucket stats - Issue here
        #     try:
        #         ratio, width, height = get_ratio_width_height(
        #             os.path.dirname(training_dirs[0])
        #         )

        #         if args.low_res_fine_tune:
        #             width, height = width // 2, height // 2

        #     except Exception as e:
        #         print(e.with_traceback)
        #         raise e
        #     # Override resolution to bucket resolution
        #     args.resolution = (height, width)
        #     if dynamic_batches:
        #         dataset_config.train_batch_size = dynamic_batches[
        #             min(dynamic_batches.keys(), key=lambda k: abs(k - height * width))
        #         ]
        #         print(
        #             f"Converting batch size to {dataset_config.train_batch_size} for resolution {width},{height}"
        #         )
        #     print(f"Chosen ratio for rank {RANK} is {training_dirs}")

        # else:
        #     height, width = args.resolution, args.resolution
        height, width = args.resolution, args.resolution

        if (
            dataset_config.dataset_name == "timm/imagenet-1k-wds"
            or dataset_config.dataset_name == "imagenet-1k"
        ):  # TODO
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(
                base_dir, "model_utils/imagenet-simple-labels.json.out"
            )
            with open(config_path) as f:
                imagenet_labels = json.load(f)
                dataset_config.image_column = dataset_name_mapping[
                    dataset_config.dataset_name
                ][0]
                dataset_config.caption_column = dataset_name_mapping[
                    dataset_config.dataset_name
                ][1]

        # 6. Get the column names for input/target.
        column_names = [dataset_config.image_column, dataset_config.caption_column]
        dataset_columns = dataset_name_mapping.get(dataset_config.dataset_name, None)
        if dataset_config.image_column is None:
            image_column = (
                dataset_columns[0] if dataset_columns is not None else column_names[0]
            )
        else:
            image_column = dataset_config.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{dataset_config.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if dataset_config.caption_column is None:
            caption_column = (
                dataset_columns[1] if dataset_columns is not None else column_names[1]
            )
        else:
            caption_column = dataset_config.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{dataset_config.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

        if args.precompute:
            if args.train_with_ratios:
                h, w = args.resolution
                res = "_".join([str(h), str(w)])
                image_column = f"latents_{res}"
            else:
                image_column = f"latents_{args.resolution}"

            column_names = [
                "prompt_embeds",
                # "pooled_prompt_embeds",
                image_column,
            ]
        else:
            column_names = [image_column, caption_column]

        dataset = None
        if args.random_latents:
            assert args.precompute == 1

            def data_generator():
                data = {}
                h, w = args.resolution
                data["prompt_embeds"] = torch.randn(
                    [args.max_sequence_length, 4096], dtype=weight_dtype
                )
                data[f"latents_{width}_{height}"] = torch.randn(
                    [16, int(h / 8), int(w / 8)], dtype=weight_dtype
                )
                while True:
                    yield data

            dataset = datasets.IterableDataset.from_generator(data_generator)
            print("Using DummyPrecompute")
        # elif dataset_config.dataset_name is not None:
        #     # Downloading and loading a dataset from the hub.
        #     if dataset_config.dataset_name == "timm/imagenet-1k-wds":
        #         token = os.environ["HF_API_TOKEN"]
        #         url = "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-train-{{0000..1023}}.tar"
        #         url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {token}'"
        #         ds = wds.WebDataset(
        #             url, nodesplitter=wds.split_by_worker, shardshuffle=True
        #         ).decode("pil")
        #         rng = random.Random(seed)
        #         dataset = ds.shuffle(10000, rng=rng)
        #         # dataset = DatasetDict({"train": ds})
        #     else:
        #         ds = load_dataset(
        #             dataset_config.dataset_name,
        #             dataset_config.dataset_config_name,
        #             streaming=True,
        #             token=True,
        #             trust_remote_code=True,
        #         )
        #         print(f"Shuffeling according to seed: {seed}")
        #         print(f"size of dataset: {len(ds)}")
        #         dataset = ds.shuffle(seed=seed, buffer_size=10_000)
        elif dataset_config.local_path:
            dataset = FinetuneDataset(
                instance_data_root=None,
                instance_prompt = None,
                class_prompt = None,
                dataset_name=dataset_config.local_path,
                image_column="image",
                caption_column="answer",
        )
        else:
            ds = load_dataset_from_tars(
                training_dirs=training_dirs,
                rank=RANK,
                world_size=WORLD_SIZE,
                seed=seed,
                slice=False,  # We don't want slicing when training with ratios
            )
            # Shuffle depends on pid (which changes from runs) and time (float time in seconds)
            rng = random.Random(seed)
            dataset = ds.shuffle(10000, rng=rng)

        transforms_todo = []
        if args.center_crop:
            print("Adding center_crop")
            transforms_todo += [
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                )
            ]
            transforms_todo += [transforms.CenterCrop(args.resolution)]
        elif args.resize:
            print("Adding resize")
            transforms_todo += [transforms.Resize(args.resolution)]

        if args.h_flip:
            print("Adding horizontal flip")
            transforms_todo += [transforms.RandomHorizontalFlip(0.5)]

        transforms_todo += [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]

        train_transforms = transforms.Compose(transforms_todo)

        def preprocess_train_webdataset(record):
            if "pickle" in record:
                record = record["pickle"]

            example = {}

            if args.precompute:
                prompt_embeds = record["prompt_embeds"].to(dtype=weight_dtype)
                seq_len, dim = prompt_embeds.shape
                # Pad to max sequence length
                if seq_len < args.max_sequence_length:
                    padding = torch.zeros(
                        (args.max_sequence_length - seq_len, dim),
                        dtype=prompt_embeds.dtype,
                        device=prompt_embeds.device,
                    )
                    prompt_embeds = torch.concat([prompt_embeds, padding], dim=0)

                example["prompt_embeds"] = prompt_embeds

                if args.train_with_ratios:
                    height, width = args.resolution
                    # h = height//8
                    # w = width//8
                    example["pixel_values"] = record[f"latents_{width}_{height}"]
                else:
                    height, width = args.resolution, args.resolution
                    # h = height//8
                    # w = width//8
                    example["pixel_values"] = record[
                        f"latents_{args.resolution}"
                    ]  # .reshape(16, h, w)

            else:
                example = preprocess_train(record)

            return example

        def preprocess_train(record):
            example = {}
            image = train_transforms(record['image'].convert("RGB"))
            caption = record[dataset_config.caption_column]

            example = {"pixel_values": image, "caption": caption}
            return example

        with accelerator.main_process_first():
            # train_dataset = dataset.map(preprocess_train)
            train_dataset = dataset

        # def collate_fn(examples):
            
        #     pixel_values = torch.stack(
        #         [example["pixel_values"] for example in examples]
        #     )
        #     pixel_values = pixel_values.to(
        #         memory_format=torch.contiguous_format
        #     ).float()

        #     if args.precompute:
        #         prompt_embeds = torch.stack(
        #             [example["prompt_embeds"] for example in examples]
        #         )
        #         # pooled_prompt_embeds = torch.stack(
        #         #     [example["pooled_prompt_embeds"] for example in examples]
        #         # )
        #         return pixel_values, prompt_embeds  # , pooled_prompt_embeds
        #     else:
        #         captions = [example["caption"] for example in examples]

        #         return pixel_values, captions

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples]
            )
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()

            captions = [example["caption"] for example in examples]

            return pixel_values, captions  

        # DataLoaders creation:
        print(f"Using {dataloader_config.num_workers} Workers")
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=dataset_config.train_batch_size,
            num_workers=dataloader_config.num_workers,
            drop_last=True,  # This is needed until we move null text to data loader
            # prefetch_factor=4,
            # pin_memory=True, #FIXME
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        if trainer_config.max_train_steps is None:
            num_update_steps_per_epoch = math.ceil(
                len(train_dataloader) / args.gradient_accumulation_steps
            )
            trainer_config.max_train_steps = (
                trainer_config.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps
            # * args.gradient_accumulation_steps
            * accelerator.num_processes,
            num_training_steps=trainer_config.max_train_steps
            * args.gradient_accumulation_steps,
        )

        # FSDP - prepare() everything else except the model
        if startegy_config.strategy_name == "fsdp":
            optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
        else:
            transformer, optimizer, lr_scheduler = accelerator.prepare(
                transformer, optimizer, lr_scheduler
            )

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            conf = {
                **vars(args),
                **{
                    "accelerator_use_distributed": accelerator.use_distributed,
                    "accelerator_mixed_precision": accelerator.mixed_precision,
                    "accelerator_distributed_type": str(accelerator.distributed_type),
                },
            }
            wandb.init(
                project=logger_config.wandb_project,
                entity=logger_config.wandb_entity,
                config=conf,
                mode=logger_config.wandb_mode,
                group=logger_config.wandb_group,
                name=logger_config.wandb_run_name,
            )
            # init wandb trackers ecce

            # accelerator.init_trackers()
            wandb.config.update(vars(args), allow_val_change=True)

        # Train!
        total_batch_size = (
            dataset_config.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        logger.info(f"diffusers version: {diffusers.__version__}")

        logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {trainer_config.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {dataset_config.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {trainer_config.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        resume_step = None
        if trainer_config.resume_from_checkpoint != "no":
            # Load from local checkpoint that sage maker synced to s3 prefix
            if trainer_config.resume_from_checkpoint != "latest":
                path = os.path.basename(trainer_config.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(trainer_config.checkpoint_local_path)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{trainer_config.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                trainer_config.resume_from_checkpoint = None
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(
                    os.path.join(trainer_config.checkpoint_local_path, path),
                    map_location="cpu",
                )
                global_step = int(path.split("_")[-1])

                first_epoch = 0  # FIXME
                resume_step = 0
                global_step += 2  # FIXME

        if args.reinit_optimizer:
            del optimizer

            if args.reinit_optimizer_type != "":
                if args.reinit_optimizer_type == "8bit_adam":
                    optimizer_cls = bnb.optim.AdamW8bit
                elif args.reinit_optimizer_type == "adamw":
                    optimizer_cls = torch.optim.AdamW
                elif args.reinit_optimizer_type == "adafactor":
                    optimizer_cls = transformers.Adafactor
                elif args.reinit_optimizer_type == "fused_adam":
                    optimizer_cls = apex.optimizers.FusedAdam
                else:
                    raise Exception("reinit optimizer type not found")

            print(
                f"replacing opt to {args.reinit_optimizer_type} with type of: {optimizer_cls}"
            )

            if optimizer_cls == transformers.Adafactor:
                print("using adafactor")
                optimizer = optimizer_cls(
                    ella.parameters(), lr=args.learning_rate, relative_step=False
                )
            elif (
                optimizer_cls == torch.optim.AdamW
            ):  # or optimizer_cls==apex.optimizers.FusedAdam:
                optimizer = optimizer_cls(
                    transformer.parameters(),
                    lr=args.learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    eps=args.adam_epsilon,
                )

            accelerator._optimizers = []
            optimizer = accelerator.prepare_optimizer(optimizer)
            print("Replacing Optimizer")

        if args.reinit_optimizer or args.reinit_scheduler:
            del lr_scheduler
            lr_scheduler = get_scheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps
                # * args.gradient_accumulation_steps
                * accelerator.num_processes,
                num_training_steps=trainer_config.max_train_steps
                * args.gradient_accumulation_steps,
            )
            accelerator._schedulers = []
            lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)
            print("Replacing LR Scheduler")

        if optimizer_cls == transformers.Adafactor:
            print("using adafactor optimizer")
        # elif optimizer_cls == apex.optimizers.FusedAdam:
        #     print('using fused adam optimizer')
        else:
            print(f"Using adam with lr: {args.learning_rate}, beta2: {args.adam_beta2}")

        # # Make sure we don't save to the same directory
        # if trainer_config.resume_from_checkpoint != "latest":
        #     args.s3_prefix += str(date.today())
        #     args.s3_prefix += (
        #         "_" + str(datetime.now().hour) + "_" + str(datetime.now().minute)
        #     )

        # TODO - used for debug
        if args.convert_unet_to_weight_dtype:
            print("Warning: Convertying models dtype directly - Remove in the future")
            transformer = transformer.to(accelerator.device, dtype=weight_dtype)

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, trainer_config.max_train_steps),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        if args.noise_offset > 0.0:
            print(f"Using noise offset {args.noise_offset}")

        now = datetime.now()
        times_arr = []

        if args.use_flow_matching:
            if args.use_dynamic_shift:
                print("init original flux schduler")
                num_inference_steps = 1000
                sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

                h, w = args.resolution
                assert h % 16 == 0 and w % 16 == 0
                image_seq_len = int(h / 16) * int(
                    w / 16
                )  # latents.shape[1] # num tokens

                mu = calculate_shift(
                    image_seq_len,
                    noise_scheduler.config.base_image_seq_len,
                    noise_scheduler.config.max_image_seq_len,
                    noise_scheduler.config.base_shift,
                    noise_scheduler.config.max_shift,
                )

                # Init sigmas and timesteps acording to shift size
                timesteps, num_inference_steps = retrieve_timesteps(
                    noise_scheduler,
                    num_inference_steps=num_inference_steps,
                    device="cuda",
                    timesteps=None,
                    sigmas=sigmas,
                    mu=mu,
                )

            def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                sigmas = noise_scheduler.sigmas.to(
                    device=accelerator.device, dtype=dtype
                )
                schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                timesteps = timesteps.to(accelerator.device)
                step_indices = [
                    (schedule_timesteps == t).nonzero().item() for t in timesteps
                ]

                assert len(step_indices) == len(timesteps)
                sigma = sigmas[step_indices].flatten()

                while len(sigma.shape) < n_dim:
                    sigma = sigma.unsqueeze(-1)
                return sigma

        if args.use_continuous_sigmas:
            print("Using smooth sigmas")

        null_conditioning = torch.zeros(
            [1, 128, 4096], dtype=weight_dtype, device=accelerator.device
        )
        print("Using zeros for null embeddings")

        assert null_conditioning.shape[0] == 1
        null_conditioning = null_conditioning.repeat(
            dataset_config.train_batch_size, 1, 1
        )

        # if args.debug and not args.precompute:
        #     text_encoder.to("cpu")

        vae_scale_factor = 2 ** len(vae.config.block_out_channels)

        for epoch in range(first_epoch, trainer_config.num_train_epochs):
            transformer.train()

            train_loss = 0.0
            iter_ = iter(train_dataloader)
            for step in range(trainer_config.max_train_steps):
                have_batch = False
                while have_batch == False:
                    try:
                        fetch_time = datetime.now()
                        batch = next(iter_)
                        fetch_time = datetime.now() - fetch_time
                        have_batch = True
                    except Exception as e:
                        if type(e) == StopIteration:
                            iter_ = iter(train_dataloader)
                            print(f"Rank {RANK} reinit iterator for {training_dirs}")
                        elif (
                            dataset_config.dataset_name == "timm/imagenet-1k-wds"
                            and type(e) == OSError
                        ):
                            iter_ = iter(train_dataloader)
                            print(
                                f"Rank {RANK} reinit iterator for {training_dirs} due to OSError"
                            )
                        elif dataset_config.dataset_name == "timm/imagenet-1k-wds":
                            iter_ = iter(train_dataloader)
                            print(
                                f"Rank {RANK} reinit iterator for {training_dirs} due to UnKnown error"
                            )
                        else:
                            raise e

                if not args.precompute:
                    pixel_values, captions = batch
                    pixel_values = pixel_values.to(device=accelerator.device)
                else:
                    (
                        pixel_values,
                        encoder_hidden_states,
                    ) = batch  # On precompute we have vae latent and t5 text embedding
                    pixel_values = pixel_values.to(device=accelerator.device)
                    encoder_hidden_states = encoder_hidden_states.to(
                        device=accelerator.device
                    )

                if (
                    trainer_config.resume_from_checkpoint != "no"
                    and epoch == first_epoch
                ):
                    if step < resume_step:
                        if step % args.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                        continue

                with accelerator.accumulate(transformer):

                    # if args.debug and not args.precompute:
                    #     transformer.to("cpu")
                    #     text_encoder.to("cpu")
                    #     vae.to("cpu")

                    if not args.precompute:
                        latents = vae.encode(
                            pixel_values
                        ).latent_dist.sample()  # 1,4,64,64
                    else:
                        latents = pixel_values

                    latents = (
                        latents - vae.config.shift_factor
                    ) * vae.config.scaling_factor

                    # Using fsdp with mixed precision requires casting the unet to dtype for ella
                    # if args.use_fsdp:
                    latents = latents.to(dtype=weight_dtype)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)

                    # Do we still need this ? with epsilon prediction probably ?
                    if args.noise_offset > 0.0:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1),
                            device=latents.device,
                        )

                    # TODO - here
                    bsz = pixel_values.shape[0]

                    if args.use_flow_matching:
                        if args.use_continuous_sigmas:
                            sigmas = compute_density_for_timestep_sampling(
                                weighting_scheme=args.weighting_scheme,
                                batch_size=bsz,
                                logit_mean=args.logit_mean,
                                logit_std=args.logit_std,
                                mode_scale=args.mode_scale,
                            )  # Give more weight to middle steps
                            timesteps = (
                                sigmas * noise_scheduler.config.num_train_timesteps
                            ).to(device=accelerator.device)
                            sigmas = sigmas.resize(
                                dataset_config.train_batch_size, 1, 1, 1
                            ).to(device=accelerator.device)
                        else:
                            u = compute_density_for_timestep_sampling(
                                weighting_scheme=args.weighting_scheme,
                                batch_size=bsz,
                                logit_mean=args.logit_mean,
                                logit_std=args.logit_std,
                                mode_scale=args.mode_scale,
                            )  # Give more weight to middle steps
                            indices = (
                                u * noise_scheduler.config.num_train_timesteps
                            ).long()
                            # I think it's related to noise shifting - taking the shifted timestep with higher noise
                            timesteps = noise_scheduler.timesteps[indices].to(
                                device=accelerator.device
                            )
                            # Add noise according to flow matching.
                            sigmas = get_sigmas(timesteps, n_dim=latents.ndim)
                            # With res 256 and shift=1 this is similar to a linear y=x schedule

                        noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
                    else:
                        # Sample a random timestep for each image
                        timesteps = torch.randint(
                            0,
                            noise_scheduler.num_train_timesteps,
                            (bsz,),
                            device=latents.device,
                        )
                        timesteps = timesteps.long()
                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(
                            latents, noise, timesteps
                        )

                    # Get text embeddings
                    if not args.precompute:

                        prompt_embeds_list = []
                        for p in captions:
                            curr_embeds = get_t5_prompt_embeds(
                                tokenizer,
                                text_encoder,
                                prompt=p,
                                max_sequence_length=args.max_sequence_length,
                            )
                            prompt_embeds_list.append(curr_embeds)
                        encoder_hidden_states = torch.cat(prompt_embeds_list, dim=0)

                    # input for rope positionak embeddings for text
                    text_ids = torch.zeros(
                        dataset_config.train_batch_size, args.max_sequence_length, 3
                    ).to(device=accelerator.device, dtype=encoder_hidden_states.dtype)
                    # Sample masks for the edit prompts.
                    if args.drop_rate_cfg > 0:
                        # null embedding for 10% of the images
                        random_p = torch.rand(
                            bsz, device=latents.device, generator=generator
                        )

                        prompt_mask = random_p < args.drop_rate_cfg
                        prompt_mask = prompt_mask.reshape(bsz, 1, 1)

                        # Final text conditioning.
                        encoder_hidden_states = torch.where(
                            prompt_mask, null_conditioning, encoder_hidden_states
                        )

                    # Get the target for loss depending on the prediction type
                    if args.use_flow_matching:
                        if args.flow_matching_latent_loss:
                            target = latents
                        else:
                            target = noise - latents  # V pred

                    elif noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )

                    # if args.debug and not args.precompute:
                    #     text_encoder.to("cpu")
                    #     vae.to("cpu")
                    #     transformer.to("cuda")

                    # Positional Embeddings input + Patchify - Consider moving to pipeline
                    # height, width = noisy_latents.shape[-2:]
                    num_channels_latents = noisy_latents.shape[1]
                    # assert num_channels_latents==16
                    # Patchify - patch=2 (4->1)
                    # Embeddings is inside the transformer (dim 16->inner dim)
                    latent_height = 2 * (int(height) // vae_scale_factor)
                    latent_width = 2 * (int(width) // vae_scale_factor)
                    patched_noisy_latents = BriaPipeline._pack_latents(
                        noisy_latents,
                        dataset_config.train_batch_size,
                        num_channels_latents,
                        latent_height,
                        latent_width,
                    )

                    # input for rope positional embeddings for latents
                    patched_latent_image_ids = BriaPipeline._prepare_latent_image_ids(
                        dataset_config.train_batch_size,
                        latent_height,
                        latent_width,
                        accelerator.device,
                        noisy_latents.dtype,
                    )

                    # t_=time.time()
                    # forward_time = datetime.now()
                    forward_time = []
                    with CudaTimerContext(forward_time):
                        model_pred = transformer(
                            hidden_states=patched_noisy_latents,  # [batch, height/patch*width/patch, 64]
                            timestep=timesteps,
                            encoder_hidden_states=encoder_hidden_states,  # [batch,128,height/patch*width/patch]
                            txt_ids=text_ids,  # [batch, 128, 3]
                            img_ids=patched_latent_image_ids,
                            return_dict=False,
                        )[0]

                        # Un-Patchify latent  (4 -> 1)
                        model_pred = BriaPipeline._unpack_latents(
                            model_pred, height, width, vae_scale_factor
                        )

                        if args.use_flow_matching:
                            if args.flow_matching_latent_loss:
                                # Get latents (x0)
                                model_pred = model_pred * (-sigmas) + noisy_latents

                            weighting = compute_loss_weighting_for_sd3(
                                weighting_scheme=args.weighting_scheme, sigmas=sigmas
                            )
                            loss = torch.mean(
                                (
                                    weighting.float()
                                    * (model_pred.float() - target.float()) ** 2
                                ).reshape(target.shape[0], -1),
                                1,
                            )
                            loss = loss.mean()
                        else:
                            # target (bs,4,64,64), encoder_hidden_states (bs,77,768), timesteps (bs), batch["pixel_values"] (bs,3,256,256)
                            loss = F.mse_loss(
                                model_pred.to(torch.float32),
                                target.to(torch.float32),
                                reduction="mean",
                            )
                    forward_time = forward_time[0]  # datetime.now() - forward_time

                    # Gather the losses across all processes for logging (if we use distributed training).
                    # Maybe? - Consider moving loss to cpu + detach before gather, since it
                    # only serves for printing total training loss metric
                    losses = accelerator.gather(
                        loss
                        # loss.repeat(dataset_config.train_batch_size)
                    )
                    avg_loss = losses.mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    # if not args.debug:
                    backward_time = []
                    with CudaTimerContext(backward_time):
                        accelerator.backward(loss)
                    backward_time = backward_time[0]

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            transformer.parameters(), args.max_grad_norm
                        )

                    # opt_step_time = datetime.now()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    # opt_step_time = datetime.now() - opt_step_time
                    # optimizer.zero_grad(set_to_none=True)  # Saves memory

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        print("train_loss: ", train_loss)
                        after = datetime.now() - now
                        now = datetime.now()

                        times_arr += [after.total_seconds()]
                        log_params = {
                            "train_loss": train_loss,
                            "num_processes": accelerator.num_processes,
                            "batch_time_sec": after.total_seconds(),
                            "lr_scheduler": lr_scheduler.get_lr()[0],
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            # "opt_step_time": opt_step_time.total_seconds(),
                            "backward_time": backward_time,
                            "forward_time": forward_time,
                            "avg_batch_time_sec_batch_": np.mean(times_arr),
                            "fetch_batch": fetch_time.total_seconds(),
                        }

                        wandb_logs = {}
                        if (
                            logger_config.save_images_to_wandb
                            and global_step % logger_config.save_images_every == 0
                        ):
                            # Log images - happens on the background - saves ~ 0.2 secs when not saving images and latent
                            wandb_logs = {
                                "latents": wandb.Image(
                                    latents.cpu(), caption="latents"
                                ),
                                "noisy_latents": wandb.Image(
                                    noisy_latents.cpu(), caption="noisy_latents"
                                ),
                                "target": wandb.Image(target.cpu(), caption="noise"),
                                "model_pred": wandb.Image(
                                    model_pred.cpu(), caption="model_pred"
                                ),
                            }
                            if not args.precompute:
                                wandb_logs["original_image"] = wandb.Image(
                                    pixel_values.cpu().to(weight_dtype),
                                    caption="original_image",
                                )
                            else:
                                wandb_logs["original_image"] = wandb.Image(
                                    vae.decode(
                                        latents.cpu() / vae.config.scaling_factor,
                                        return_dict=False,
                                    )[0]
                                    .clamp(-1, 1)
                                    .to(weight_dtype),
                                    caption="original_image",
                                )

                        # print('Dumping logs to wandb')
                        # t=time.time()
                        wandb.log(
                            {**log_params, **wandb_logs},
                            step=global_step,
                        )
                        # print('wandb',time.time()-t)

                    train_loss = 0.0

                if (global_step - 1) % trainer_config.checkpointing_steps == 0 and (
                    global_step - 1
                ) > 0:
                    save_path = os.path.join(
                        trainer_config.checkpoint_local_path,
                        f"checkpoint_{global_step - 1}",
                    )
                    # saved by process 0 on fsdp
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    # Clear memory after save
                    # torch.cuda.empty_cache()

                logs = {"step_loss": loss.detach().item()}

                progress_bar.set_postfix(**logs)
                if global_step >= trainer_config.max_train_steps:
                    break

        # Create the pipeline using the trained modules and save it.
        print("Waiting for everyone :)")
        accelerator.wait_for_everyone()
        accelerator.end_training()
