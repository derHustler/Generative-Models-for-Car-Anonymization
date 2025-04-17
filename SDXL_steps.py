#!/usr/bin/env python3
"""Cityscapes Multi‑Car Inpainting Script (aufgeräumte Zeilen, Logik unverändert)"""

# -------------------------------------------------------------
# Farb‑Escape‑Sequenzen & Zeitstempel
# -------------------------------------------------------------
from datetime import datetime
RED = "\033[91m"
WHITE = "\033[0m"

now = datetime.now()
print(f"Aktuelles Datum und Uhrzeit: {RED}{now.strftime('%Y-%m-%d %H:%M:%S')}{WHITE}")

# -------------------------------------------------------------
# Standard‑Bibliotheken
# -------------------------------------------------------------
import argparse
import json
import os
import sys
import time

# Externe Libraries
import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image 

# -------------------------------------------------------------
# System‑Infos
# -------------------------------------------------------------
print("GPU COUNT", torch.cuda.device_count())
print("Current Python Environment:", sys.prefix)

# -------------------------------------------------------------
# Argument‑Parsing
# -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str)
parser.add_argument("--input_dir", type=str)  # Hier liegen die Bilder & Masken aus maskgenerator.py
args = parser.parse_args()

image_output_dir = args.output_dir
base_dir = args.input_dir

cities = ["frankfurt", "lindau", "munster"]  # ggf. anpassen

print(f"image_output_dir: {RED}'{image_output_dir}'{WHITE}.")

# -------------------------------------------------------------
# Verzeichnis‑Anlage
# -------------------------------------------------------------
if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)
    for city in cities:
        os.makedirs(os.path.join(image_output_dir, city))
    print(f"Verzeichnis '{image_output_dir}' und Unterordner für Städte wurden erstellt.")
else:
    print(f"Verzeichnis '{image_output_dir}' existiert bereits.")

# -------------------------------------------------------------
# Bild‑Basenamen sammeln
# -------------------------------------------------------------
pictures: dict[str, list[str]] = {}
counter_pics = 0

for city in cities:
    city_dir = os.path.join(base_dir, "images", city)
    pictures[city] = []

    for filename in os.listdir(city_dir):
        counter_pics += 1
        if filename.endswith("_leftImg8bit.png"):
            base_name = filename.replace("_leftImg8bit.png", "")
            pictures[city].append(base_name)

print(f"{counter_pics} Bilder geladen")

# -------------------------------------------------------------
# Diffusion‑Pipeline
# -------------------------------------------------------------
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
).to("cuda")

prompt = (
    "A photorealistic car seamlessly integrated into an urban street scene, with consistent texture and lighting. "
    "Replace the original car with a modern, neutral design, ensuring no logos, license plates, or text. The new car "
    "should have a generic color, blending naturally with realistic reflections and shadows."
)
negative_prompt = (
    "worst quality, low resolution, overexposed, blurry, distorted shapes, numbers on doors, text artifacts, "
    "unrealistic shadows."
)

generator = torch.Generator(device="cuda").manual_seed(42)

# -------------------------------------------------------------
# Haupt‑Loop: Inpainting je City & Bild
# -------------------------------------------------------------
counter_pics = 0  # erneutes Counting für Verarbeitung
zeit_total = 0.0

for city in cities:
    for base_name in pictures[city]:
        json_path = os.path.join(base_dir, "masks", city, f"{base_name}_gtFine_polygons.json")
        image_path = os.path.join(base_dir, "images", city, f"{base_name}_leftImg8bit.png")

        in_image = load_image(image_path)

        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Alle "car"‑Polygone extrahieren
        polygons = [
            np.array(obj["polygon"], np.int32).reshape((-1, 1, 2))
            for obj in results["objects"]
            if obj["label"] == "car"
        ]

        # Originalbild & Masken vorbereiten
        gt_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = np.zeros((1024, 2048), dtype=np.uint8)

        time_pic = 0.0
        image_before = in_image
        car_counter = 0

        for polygon in polygons:
            cv2.fillPoly(mask, [polygon], 255)
            in_mask = Image.fromarray(mask)

            start = time.time()
            result = pipe(
                prompt=prompt,
                image=image_before,
                mask_image=in_mask,
                height=1024,
                width=2048,
                guidance_scale=6.0,
                generator=generator,
                num_inference_steps=25,
                negative_prompt=negative_prompt,
                strength=0.999,
            ).images[0]
            time_pic += time.time() - start

            result_np = np.array(result)
            car_counter += 1
            Image.fromarray(result_np).save(
                os.path.join(image_output_dir, city, f"{base_name}_{car_counter}.png")
            )

            # Ergebnis mit Original verschmelzen
            masked_result = cv2.bitwise_and(result_np, result_np, mask=mask)
            inverse_mask = cv2.bitwise_not(mask)
            gt_background = cv2.bitwise_and(gt_rgb, gt_rgb, mask=inverse_mask)
            gt_rgb = cv2.add(gt_background, masked_result)

            image_before = Image.fromarray(gt_rgb)
            mask.fill(0)  # Maske zurücksetzen
            counter_pics += 1

        zeit_total += time_pic
        out_norm = f"{base_name}.png"
        Image.fromarray(gt_rgb).save(os.path.join(image_output_dir, city, out_norm))

        print(f"Image saved to: {os.path.join(image_output_dir, city, out_norm)}")
        print(f"Verstrichene Zeit: {zeit_total:.2f} Sekunden für {counter_pics} Autos")

print("Ende")
