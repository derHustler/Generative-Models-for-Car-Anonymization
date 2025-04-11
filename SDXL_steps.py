from datetime import datetime
RED = "\033[91m"
WHITE = "\033[0m"
# Aktuelles Datum und Uhrzeit im benutzerdefinierten Format
now = datetime.now()
print(f"Aktuelles Datum und Uhrzeit: {RED}{now.strftime('%Y-%m-%d %H:%M:%S')}{WHITE}")

import torch
#accelerator = Accelerator()
print("GPU COUNT", torch.cuda.device_count())



import sys
print("Current Python Environment:", sys.prefix)
index = "failedIndex"
import cv2
import numpy as np
import json
import os
from PIL import Image

import time
#from accelerate import Accelerator


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--i", type=str)
args=parser.parse_args()
index = args.i



base_dir="/work/rn583pgoa-workdir/dataTry2"
cities = ['frankfurt', 'lindau', 'munster']


# Zielverzeichnisse für Bilder und Masken
image_output_dir = f"/work/rn583pgoa-workdir/generatedImages/steps/SDXL_{index}"
combined_dir = f"/work/rn583pgoa-workdir/generatedImages/steps/combined_{index}"

print(f"image_output_dir: {RED}'{image_output_dir}'{WHITE}.")

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


from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, check_min_version

pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16
).to("cuda")
#if torch.cuda.device_count() > 1:
  # pipe = accelerator.prepare(pipe)



prompt = "A photorealistic car seamlessly integrated into an urban street scene, with consistent texture and lighting. Replace the original car with a modern, neutral design, ensuring no logos, license plates, or text. The new car should have a generic color, blending naturally with realistic reflections and shadows."
#prompt = "Photorealistic car in Street"
generator = torch.Generator(device="cuda").manual_seed(42)
negative_prompt="worst quality, low resolution, overexposed, blurry, distorted shapes, numbers on doors, text artifacts, unrealistic shadows."
#negative_prompt="trash"


import shutil
import matplotlib.pyplot as plt
counter_pics = 0
time_glob = 0

for city in cities:
        
    for base_name in pictures[city]:

        json_path = os.path.join(base_dir, "masks", city, f"{base_name}_gtFine_polygons.json")
        image_path = os.path.join(base_dir, "images" , city, f"{base_name}_leftImg8bit.png")           
  
        in_image = load_image(
            image_path
        )

        with open(json_path, 'r') as f:
            results = json.load(f)

        all_polygons = []
        
        for item in results["objects"]:
            if item["label"] == "car":
                polygon_points = np.array(item['polygon'], np.int32)
                polygon_points = polygon_points.reshape((-1, 1, 2))
                all_polygons.append(polygon_points)

        gt_bgr = cv2.imread(image_path)
        gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        
        mask = np.zeros((1024, 2048), dtype=np.uint8)

        counter=0
        time_pic = 0

        image_before = in_image
        
        for polygon in all_polygons:
            cv2.fillPoly(mask, [polygon], 255)
            in_mask = Image.fromarray(mask)
            start = time.time()
        
            result = pipe(
                prompt = prompt,
                image = image_before,
                mask_image = in_mask,
                height=1024,
                width=2048,
                guidance_scale = 6.0,
                generator= generator,
                num_inference_steps= 25,
                negative_prompt = negative_prompt,
                strength = 0.999
            ).images[0]
            end=time.time()
            time_it = end - start
            counter = counter +1
            time_pic = time_pic + time_it

            result_np = np.array(result)
            Image.fromarray(result_np).save(os.path.join(image_output_dir, city, f"{base_name}_{counter}.png"))
       
            masked_result = cv2.bitwise_and(result_np, result_np, mask=mask)
        
            inverse_mask = cv2.bitwise_not(mask)
        
            gt_background = cv2.bitwise_and(gt, gt, mask=inverse_mask)
        
            gt = cv2.add(gt_background, masked_result)

            image_before=Image.fromarray(gt)
        
            mask = np.zeros((1024, 2048), dtype=np.uint8)

            counter_pics = counter_pics + 1

        time_glob=time_glob + time_pic    
           
        counter=0

        out_pic=f"{base_name}_SDXL.png"
        out_norm =f"{base_name}.png"
        
        Image.fromarray(gt).save(os.path.join(image_output_dir, city, out_norm))
        Image.fromarray(gt).save(os.path.join(combined_dir, city, out_pic))
        #in_image.save(os.path.join(combined_dir, city, out_norm))
        
        
        
        print(f"Image saved to: {os.path.join(image_output_dir, city, out_norm)}")
        print(f"Image saved to: {os.path.join(combined_dir, city, out_pic)}")
        
        print(f"Verstrichene Zeit: {time_glob:.2f} Sekunden für {counter_pics} Autos")


print("Ende")

