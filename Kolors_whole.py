from datetime import datetime

# Aktuelles Datum und Uhrzeit im benutzerdefinierten Format
now = datetime.now()
print("Aktuelles Datum und Uhrzeit:", now.strftime("%Y-%m-%d %H:%M:%S"))

import torch
print("GPU COUNT", torch.cuda.device_count())

import sys
print("Current Python Environment:", sys.prefix)
index = "failedIndex"
import random

import cv2
import numpy as np
import json
import os
from PIL import Image
#from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)
from diffusers.utils import load_image, check_min_version
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import StableDiffusionXLInpaintPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer


#accelerator = Accelerator()

# Basispfad zu den Cityscapes-Daten
base_dir="/work/rn583pgoa-workdir/data"
cities = ['frankfurt', 'lindau', 'munster']

# Zielverzeichnisse für Bilder und Masken

# Array, um die Dateinamen zu speichern
pictures = {}

 
counter_pics = 0
for city in cities:
    city_dir = os.path.join(base_dir, "images", city)
    
    pictures[city] = []
    for filename in os.listdir(city_dir):
        counter_pics=counter_pics+1
        if filename.endswith("_leftImg8bit.png"):
            base_name = filename.split("_leftImg8bit.png")[0]
            pictures[city].append(base_name)
print(f"{counter_pics} Bilder geladen")

# Ausgabe der gespeicherten Dateinamen
#print(pictures)


ckpt_dir = "/work/rn583pgoa-workdir/Kolors/weights/Kolors-Inpainting"
text_encoder = ChatGLMModel.from_pretrained(
    f'{ckpt_dir}/text_encoder',
    torch_dtype=torch.float16).half()
tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()

pipe = StableDiffusionXLInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler
).to("cuda")

#if torch.cuda.device_count() > 1:
#    pipe = accelerator.prepare(pipe)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--i", type=str)
args=parser.parse_args()
index = args.i

# Zielverzeichnisse für Bilder und Masken
image_output_dir = f"/work/rn583pgoa-workdir/generatedImages/whole/Kolors_{index}"
combined_dir = f"/work/rn583pgoa-workdir/generatedImages/whole/combined_{index}"

# Prüfen, ob das Verzeichnis existiert
if not os.path.exists(image_output_dir):
    # Verzeichnis erstellen
    os.makedirs(image_output_dir)
    for city in cities:
        os.makedirs(os.path.join(image_output_dir, city))
    print(f"Verzeichnis '{image_output_dir}' und Unterordner für Städte wurden erstellt.")
else:
    print(f"Verzeichnis '{image_output_dir}' existiert bereits.")


if not os.path.exists(combined_dir):
    # Verzeichnis erstellen
    os.makedirs(combined_dir)
    for city in cities:
        os.makedirs(os.path.join(combined_dir, city))
    print(f"Verzeichnis '{combined_dir}' und Unterordner für Städte wurden erstellt.")

else:
    print(f"Verzeichnis '{combined_dir}' existiert bereits.")

prompt = "A photorealistic car seamlessly integrated into an urban street scene, with consistent texture and lighting. Replace the original car with a modern, neutral design, ensuring no logos, license plates, or text. The new car should have a generic color, blending naturally with realistic reflections and shadows."
generator = torch.Generator(device="cpu").manual_seed(42)
negative_prompt="worst quality, low resolution, overexposed, blurry, distorted shapes, numbers on doors, text artifacts, unrealistic shadows."



import shutil
import matplotlib.pyplot as plt
import time

time_glob = 0
counter_pics = 0
for city in cities:
    
    
    for base_name in pictures[city]:

        # Lade das Bild und die zugehörige JSON-Datei
        json_path = os.path.join(base_dir, "masks", city, f"{base_name}_gtFine_polygons.json")
        image_path = os.path.join(base_dir, "images" , city, f"{base_name}_leftImg8bit.png")           
  
        in_image = load_image(
            image_path
        )

        with open(json_path, 'r') as f:
            results = json.load(f)

        all_polygons = []
        
        # Iteriere über alle Objekte und extrahiere die Polygone für Autos
        for item in results["objects"]:
            if item["label"] == "car":
                polygon_points = np.array(item['polygon'], np.int32)
                polygon_points = polygon_points.reshape((-1, 1, 2))
                all_polygons.append(polygon_points)

        gt_bgr = cv2.imread(image_path)
        gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

        mask = np.zeros((1024, 2048), dtype=np.uint8)

        time_pic = 0
        
        counter=0
        for polygon in all_polygons:
            cv2.fillPoly(mask, [polygon], 255)
            
        in_mask = Image.fromarray(mask)
        
        
        start = time.time()
        result = pipe(
            prompt = prompt,
            image = in_image,
            mask_image = in_mask,
            height=1024,
            width=2048,
            guidance_scale = 6.0,
            generator= generator,
            num_inference_steps= 25,
            negative_prompt = negative_prompt,
            num_images_per_prompt = 1,
            strength = 0.999
        ).images[0]

        end=time.time()

        time_pic= end - start
        
        result_np = np.array(result)
    
        masked_result = cv2.bitwise_and(result_np, result_np, mask=mask)
    
        inverse_mask = cv2.bitwise_not(mask)
    
        gt_background = cv2.bitwise_and(gt, gt, mask=inverse_mask)
        
        gt = cv2.add(gt_background, masked_result)

        mask = np.zeros((1024, 2048), dtype=np.uint8)

        counter_pics = counter_pics + 1
        
        
        time_glob=time_glob + time_pic    
        counter=0
        
        out_pic=f"{base_name}_Kolors.png"
        out_norm =f"{base_name}.png"
        out_OG=f"{base_name}_O.png"

        
        Image.fromarray(gt).save(os.path.join(image_output_dir, city, out_norm))
        Image.fromarray(gt).save(os.path.join(combined_dir, city, out_pic))
        in_image.save(os.path.join(combined_dir, city, out_OG))
       
        print(f"Image saved to: {os.path.join(image_output_dir, city, out_norm)}")
        print(f"Image saved to: {os.path.join(combined_dir, city, out_pic)}")
        
        print(f"Verstrichene Zeit: {time_glob:.2f} Sekunden für {counter_pics} Bilder")








print("Ende")