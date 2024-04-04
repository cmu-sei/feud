#!/usr/bin/env python3

# ======================================================================================================================
# SaTML CNN Interpretability Competition Submission

# Copyright 2024 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" 
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER 
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED 
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM 
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see 
# Copyright notice for non-US Government use and distribution.

# This Software includes and/or makes use of Third-Party Software each subject to its own license.

# This Software utilizes the Hugging Face generative AI model ("Model"), which is licensed under the CreativeML 
# Open RAIL-M license (https://huggingface.co/spaces/CompVis/stable-diffusion-license). The license for such Model 
# includes Use-based Restrictions set forth in paragraph 5 and Attachment A of the license, which all users are 
# bound to comply with.

# DM24-0211
# ======================================================================================================================

# ======================================================================
# https://huggingface.co/spaces/anzorq/finetuned_diffusion
# https://huggingface.co/spaces/pharmapsychotic/CLIP-Interrogator
# ======================================================================

# ======================================================================
# Hayden Moore, Carnegie Mellon University, SEI, AI Division
# David Shriver, Carnegie Mellon University, SEI, AI Division
# Additional Contributors: Marissa Connor, Keltin Grimes
# SaTML CNN Interpretability Competition
# 2nd IEEE Conference on Secure and Trustworthy Machine Learning (2024)
# =====================================================================

import argparse
import dataclasses
import logging
import pathlib
import random

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from clip_interrogator import Config, Interrogator
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

# Default ImageNet transforms
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

@dataclasses.dataclass
class TriggerConfig:
    model_path: pathlib.Path
    data_path: pathlib.Path
    trigger_output_path: pathlib.Path

    target_class: int
    source_class: int = None

    initial_trigger_path: pathlib.Path  = None
    trigger_size: tuple = (3, 64, 64)
    trigger_color: float = 0.5

    learning_rate: float = 4e-3
    batch_size: int = 64
    num_iterations: int = 128
    num_batches: int = 1

    seed: int = 0

    cpu: bool = False
    debug: bool = False
    log_filepath: pathlib.Path = pathlib.Path("trigger_recovery.log")


def _size_type(value: str) -> tuple:
    return tuple(int(v.strip()) for v in value.split(","))


def parse_args(args: list) -> TriggerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=pathlib.Path)
    parser.add_argument("-D", "--dataset", type=pathlib.Path, default="/dataroot/ImageNet/train", dest="data_path")

    parser.add_argument("-T", "--target", type=int, dest="target_class", required=True)
    parser.add_argument("-S", "--source", type=int, dest="source_class")

    parser.add_argument("--initial-trigger", type=pathlib.Path, dest="initial_trigger_path")
    parser.add_argument("--trigger-size", type=_size_type, default=(3, 64, 64))
    parser.add_argument("--trigger-color", type=float, default=0.5)

    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-bs", "--batch-size", type=int, default=1)
    parser.add_argument("-I", "--num-iterations", type=int, default=1000)
    parser.add_argument("-N", "--num-batches", type=int, default=1)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--logfile", type=pathlib.Path, default=pathlib.Path("trigger_recovery.log"), dest="log_filepath"
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "-o", "--output", type=pathlib.Path, default=pathlib.Path("trigger.png"), dest="trigger_output_path"
    )
    return TriggerConfig(**vars(parser.parse_args(args)))


class RecoverTrigger(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        trigger_size: tuple = (3, 64, 64),
        trigger_color: float = 0.5,
        trigger_output_path: pathlib.Path = "trigger.png",
        initial: torch.FloatTensor = None,
        trigger_transform=None,
        lr: float = 4e-3,
    ):
        super().__init__()
        # Freeze the poisoned model in eval model
        self.model = torch.jit.freeze(torch.jit.script(model.eval()))
        self.trigger_color = trigger_color

        # Check if we are loading from a trained trigger or from scratch
        if initial != None:
            self.trigger = nn.Parameter(initial.clone())
        else:
            if self.trigger_color >= 0:
                # Use user defined color starting point
                self.trigger = nn.Parameter(torch.full(trigger_size, self.trigger_color))
            else:
                # Random starting point
                self.trigger = nn.Parameter(torch.rand(trigger_size))
        
        # Set the desired size of the trigger
        self.trigger_size = trigger_size

        self.trigger_output_path = trigger_output_path

        # Set trigger transform
        self.trigger_transform = trigger_transform if trigger_transform else lambda x: x

        # Set up Adam optimizer
        self.optimizer = optim.Adam([self.trigger], lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Insert the trigger into the batch and send it through the model

        # Clamp trigger between [0,1]
        trigger = torch.clamp(self.trigger, 0, 1)

        # Pick a random position for each image in the batch
        trigger_height, trigger_width = self.trigger_size[-2:]
        trigger_start_h = torch.randint(0, x.size(2) - trigger_height, (x.size(0),))
        trigger_start_w = torch.randint(0, x.size(3) - trigger_width, (x.size(0),))

        # Apply the trigger to each input
        x_with_trigger = x.clone()
        for i, (start_h, start_w) in enumerate(zip(trigger_start_h, trigger_start_w)):
            x_with_trigger[i, :, start_h : start_h + trigger_height, start_w : start_w + trigger_width] = (
                self.trigger_transform(trigger)
            )

        # Pass Augmented Trigger + Input Image through Poisoned Model
        return self.model(x_with_trigger)

    def optimize_trigger(self, x: torch.Tensor, target_class: int, num_iterations: int = 1000, device: str = "cuda") -> torch.Tensor:
        self.train()
        self.model.eval()
        for i in range(num_iterations):
            self.optimizer.zero_grad()
            logits = self(x)
            img_pth = "images/" + str(target_class) + "/" + str(random.randint(1,5)) + ".png"
            embedd_img = Image.open(img_pth)
            loss = (
                    -F.log_softmax(logits, dim=1)[:, target_class].mean()
                    + 1e-4 * kornia.losses.total_variation(self.trigger).mean()
                    + 1e-2 * (1 - (2 * torch.clamp(self.trigger, 0, 1) - 1).abs()).mean()
                )
            
            # Resize 
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])

            dataset_embedding = transform(embedd_img)

            similarity_loss = F.cosine_similarity(torch.clamp(self.trigger, 0, 1).unsqueeze(0).to(device), 
                                                      dataset_embedding.unsqueeze(0).to(device), dim=1)
            similarity_loss = similarity_loss.mean()

            # More dramatic changes to trigger based on similarity
            if similarity_loss > 0:
                loss = loss * (random.randint(1,3))
            else:
                loss = loss * 0
            
            # Less dramatic, subtract cosine similarity from the total loss
            # loss -= (0.0025 * similarity_loss)
                
            loss.backward()
            self.optimizer.step()
        logging.info(f"loss={loss.item()} ; accuracy={(logits.argmax(dim=1) == target_class).float().mean().item()}")
        self.eval()

        # Return the optimized trigger
        return torch.clamp(self.trigger.detach().cpu(), 0, 1)

    def img_to_txt(self) -> str:
        MODELS = ['ViT-L (best for Stable Diffusion 1.*)']
        clip_config = Config(clip_model_name="ViT-L-14/openai")
        ci = Interrogator(clip_config)
        ci.config.blip_num_beams = 64
        ci.config.chunk_size = 2048
        ci.config.flavor_intermediate_count = 2048

        image = Image.open(self.trigger_output_path)
        image = image.convert('RGB')
        
        prompt = ci.interrogate(image)

        return prompt.split(',')[0]

    def img_to_img(self, prompt, trigger, strength=0.5, device="cuda") -> Image:
        generator = torch.Generator(device).manual_seed(0)
        
        # Get OpenJourneyv4 Diffusion Model name information
        model_name, model_path, model_prefix = "Midjourney v4 style", "prompthero/midjourney-v4-diffusion", "mdjrny-v4 style "
        
        # Setup Img to Img Diffusion Model with OpenJourneyv4
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler"))
        pipe = pipe.to(device)
        
        # Disable Safety Checker
        pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

        # Combine Prompt with OpenJourney's prefix
        prompt = model_prefix + prompt
        img = Image.open(trigger)
        ratio = min(512 / img.height, 512 / img.width)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        result = pipe(
            prompt,
            negative_prompt = "",
            num_images_per_prompt=1,
            image = img,
            num_inference_steps = 25,
            strength = strength,
            guidance_scale = 7.5,
            generator = generator,
            callback=None)

        return result

def log_progress(prefix: str, loader_len: int, i: int, log_at_proportions: list = [0.05, 0.25, 0.50, 0.75]):
    if any(i == int(p * loader_len) for p in log_at_proportions):
        logging.info("%s %d/%d", prefix, i, loader_len)


def main(args: list = None):
    config = parse_args(args)

    # Set users device, check is CUDA is avaliable otherwise default to cpu
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")

    # Setup file handler, stream handler, and basic logging
    file_handler = logging.FileHandler(config.log_filepath)
    stream_handler = logging.StreamHandler()
    logging.basicConfig(
        handlers=[file_handler, stream_handler],
        level=logging.DEBUG if config.debug else logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    # Set random seeds
    logging.info("Setting random seed to %d", config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    logging.info("Using device: %s", device)

    logging.info("Recovering trigger for source class: %s", config.source_class)
    logging.info("Recovering trigger for target class: %d", config.target_class)

    logging.info("Setting up transforms")

    normalize = transforms.Normalize(mean=MEAN, std=STD)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    trigger_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize(config.trigger_size[-2:], antialias=None),
            normalize,
        ]
    )

    logging.info("Loading data")
    train_dataset = ImageFolder(root=config.data_path, transform=transform)
    # filter out the target class or filter to a source class
    train_dataset.samples = [
        sample
        for sample in train_dataset.samples
        if sample[1] != config.target_class and (config.source_class is None or sample[1] == config.source_class)
    ]
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=10)

    # Load the trained model
    logging.info("Loading model")
    poisoned_model = resnet50(num_classes=1000)
    poisoned_model.load_state_dict(torch.load(config.model_path))
    poisoned_model.to(device)
    poisoned_model.eval()

    trigger_optimizer = RecoverTrigger(
        poisoned_model,
        trigger_size=config.trigger_size,
        trigger_color=config.trigger_color,
        initial=(
            None
            if config.initial_trigger_path is None
            else to_tensor(
                Image.open(config.initial_trigger_path).resize(config.trigger_size[-2:], Image.Resampling.NEAREST)
            )
        ),
        trigger_transform=trigger_transform,
        lr=config.learning_rate,
    ).to(device)

    logging.info("Training trigger")
    trigger_output_path = config.trigger_output_path
    trigger_output_path.parent.mkdir(exist_ok=True, parents=True)
    
    for i, (inputs, _) in enumerate(train_loader):
        log_progress("iteration", len(train_loader), i)
        # Send data to device
        inputs = inputs.to(device)

        # Optimize the trigger
        recovered_trigger = trigger_optimizer.optimize_trigger(inputs, config.target_class, config.num_iterations, device=device)

        # Save the debug trigger
        save_image(recovered_trigger, "debug/trigger_debug_" + str(i) + ".png")
        save_image(recovered_trigger, "debug/trigger_debug.png")
        # Save the trigger to user specified area
        save_image(recovered_trigger, config.trigger_output_path)
        
        # Break after number of defined batches has been seen by the trigger
        if i >= config.num_batches:
            break
    
    # Generate prompt from optimized trigger
    recovered_prompt = trigger_optimizer.img_to_txt()
    logging.info(f"Recovered Prompt: {recovered_prompt}")

    # Generate 10 images
    strength = 0.5
    for i in range(10):
        # Diffuse(optimized trigger + prompt)
        diffused_trigger = trigger_optimizer.img_to_img(recovered_prompt, config.trigger_output_path, strength=strength, device=device)
        # Save the trigger
        transform = transforms.Compose([transforms.ToTensor()])
        tensor = transform(diffused_trigger.images[0])
        save_image(tensor, "trigger_final_" + str(i) + ".png")
        strength += 0.025
    

if __name__ == "__main__":
    main()

