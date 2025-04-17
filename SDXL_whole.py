#!/usr/bin/env python3

"""Cityscapes‑Car‑Inpainting – Zeilen aufgeräumt (Logik unverändert)"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
from datetime import datetime
import time
import sys
import random
import json
import os
import argparse
import shutil

import cv2
import numpy as np
from PIL import Image

import torch
# from accelerate import Accelerator
import matplotlib.pyplot as plt

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, check_min_version  # noqa: F401  (check_min_version bleibt ungenutzt)

# -------------------------------------------------------------
# Laufzeit‑Infos
# -------------------------------------------------------------
now = datetime.now()
print("Aktuelles Datum und Uhrzeit:", now.strftime("%Y-%m-%d %H:%M:%S"))

print("GPU COUNT", torch.cuda.device_count())
print("Current Python Environment:", sys.prefix)
# accelerator = Accelerator()

# -------------------------------------------------------------
# Argument‑Parsing
# -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str)
parser.add_argument("--input_dir", type=str)  # Hier liegen die Bilder & Masken aus maskgenerator.py
args = parser.parse_args()

# -------------------------------------------------------------
# Verzeichnis‑Setup
# -------------------------------------------------------------
image_output_dir = args.output_dir
base_dir = args.input_dir

cities = ["frankfurt", "lindau", "munster"]  # ggf. weitere Cityscapes‑Städte eintragen

if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)
    for city in cities:
        os.makedirs(os.path.join(image_output_dir, city))
    print(f"Verzeichnis '{image_output_dir}' und Unterordner für Städte wurden erstellt.")
else:
    print(f"Verzeichnis '{image_output_dir}' existiert bereits.")

# -------------------------------------------------------------
# Bild‑ und Masken‑Listen aufbauen
# -------------------------------------------------------------
pictures = {}
counter_pics = 0

for city in cities:
    city_dir = os.path.join(base_dir, "images", city)
    pictures[city] = []

    for filename in os.listdir(city_dir):
        counter_pics += 1
        if filename.endswith("_leftImg8bit.png"):
            base_name = filename.split("_leftImg8bit.png")[0]
            pictures[city].append(base_name)

print(f"{counter_pics} Bilder geladen")

# -------------------------------------------------------------
# Diffusion‑Pipeline initialisieren
# -------------------------------------------------------------
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
).to("cuda")

prompt = (
    "A photorealistic car seamlessly integrated into an urban street scene, "
    "with consistent texture and lighting. Replace the original car with a modern, "
    "neutral design, ensuring no logos, license plates, or text. The new car should "
    "have a generic color, blending naturally with realistic reflections and shadows."
)
negative_prompt = "worst quality, low resolution, overexposed, blurry, distorted shape"

generator = torch.Generator(device="cuda").manual_seed(42)

# -------------------------------------------------------------
# Haupt‑Schleife – Inpainting pro Bild
# -------------------------------------------------------------
counter_pics = 0  # Zähler neu starten für Bearbeitung
time_glob = 0.0

for city in cities:
    for base_name in pictures[city]:
        json_path = os.path.join(base_dir, "masks", city, f"{base_name}_gtFine_polygons.json")
        image_path = os.path.join(base_dir, "images", city, f"{base_name}_leftImg8bit.png")

        # Originalbild und Annotationen laden
        in_image = load_image(image_path)
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Alle *car*‑Polygone sammeln
        all_polygons = []
        for item in results["objects"]:
            if item["label"] == "car":
                polygon_points = np.array(item["polygon"], np.int32).reshape((-1, 1, 2))
                all_polygons.append(polygon_points)

        # Bild & Masken vorbereiten
        gt_bgr = cv2.imread(image_path)
        gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        mask = np.zeros((1024, 2048), dtype=np.uint8)
        for polygon in all_polygons:
            cv2.fillPoly(mask, [polygon], 255)
        in_mask = Image.fromarray(mask)

        # Inpainting ausführen
        start = time.time()
        result = pipe(
            prompt=prompt,
            image=in_image,
            mask_image=in_mask,
            height=1024,
            width=2048,
            guidance_scale=6.0,
            generator=generator,
            num_inference_steps=25,
            negative_prompt=negative_prompt,
            strength=0.999,
        ).images[0]
        time_pic = time.time() - start

        # Ergebnis in Originalbild einfügen
        result_np = np.array(result)
        masked_result = cv2.bitwise_and(result_np, result_np, mask=mask)
        inverse_mask = cv2.bitwise_not(mask)
        gt_background = cv2.bitwise_and(gt, gt, mask=inverse_mask)
        gt = cv2.add(gt_background, masked_result)

        # Reset (unverändert zur Original‑Version)
        mask = np.zeros((1024, 2048), dtype=np.uint8)

        counter_pics += 1
        time_glob += time_pic

        out_norm = f"{base_name}.png"

        Image.fromarray(gt).save(os.path.join(image_output_dir, city, out_norm))

        print(f"Image saved to: {os.path.join(image_output_dir, city, out_norm)}")
        print(f"Verstrichene Zeit: {time_glob:.2f} Sekunden für {counter_pics} Bilder")

print("Ende")
