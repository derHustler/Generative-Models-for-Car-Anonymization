#!/usr/bin/env python3


# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import argparse
import os

import cv2
import numpy as np
from PIL import Image

# -------------------------------------------------------------
# Argument‑Parsing
# -------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Fügt GAN‑generierte Autos mithilfe binärer Masken in Originalbilder ein.",
)
parser.add_argument("--ganpics", type=str, help="Pfad zu generierten PNG‑Bildern (512×256)")
parser.add_argument("--originalpics", type=str, help="Pfad zu Original‑Bildern (2048×1024)")
parser.add_argument("--masks", type=str, help="Pfad zu binären Masken (2048×1024)")
parser.add_argument("--output", type=str, help="Zielordner für gemergte Bilder")
args = parser.parse_args()

genPth = args.ganpics
ogPth = args.originalpics
maPth = args.masks
ouPth = args.output

print(f"Generierte Bilder aus: {genPth}")
print(f"Original Bilder aus: {ogPth}")
print(f"Masken aus: {maPth}")
print(f"Output Bilder nach: {ouPth}")

# -------------------------------------------------------------
# Basename‑Sammlung
# -------------------------------------------------------------
base_names: list[str] = []
for filename in os.listdir(ogPth):
    if filename.endswith("_leftImg8bit.png"):
        base_names.append(filename.replace("_leftImg8bit.png", ""))
print("Bilder eingelesen:", len(base_names))

# -------------------------------------------------------------
# Haupt‑Loop
# -------------------------------------------------------------
for base_name in base_names:
    # Dateinamen zusammensetzen
    mask_file = f"{base_name}_gtFine_binary.png"
    img_file = f"{base_name}_leftImg8bit.png"

    # Dateien laden
    mask_np = cv2.imread(os.path.join(maPth, mask_file), cv2.IMREAD_GRAYSCALE)
    og_np_bgr = cv2.imread(os.path.join(ogPth, img_file))
    og_np = cv2.cvtColor(og_np_bgr, cv2.COLOR_BGR2RGB)

    gen_np_bgr = cv2.imread(os.path.join(genPth, img_file))  # 512×256
    gen_np = cv2.cvtColor(gen_np_bgr, cv2.COLOR_BGR2RGB)
    gen_np = cv2.resize(gen_np, (2048, 1024), interpolation=cv2.INTER_CUBIC)

    # Maske anwenden
    gen_cars = cv2.bitwise_and(gen_np, gen_np, mask=mask_np)
    inverse_mask = cv2.bitwise_not(mask_np)
    og_background = cv2.bitwise_and(og_np, og_np, mask=inverse_mask)
    final = cv2.add(og_background, gen_cars)

    # Speichern
    os.makedirs(ouPth, exist_ok=True)
    output_name = f"{base_name}_merged.png"
    Image.fromarray(final).save(os.path.join(ouPth, output_name))
    print(f"Bild gespeichert: {os.path.join(ouPth, output_name)}")

print("Ende")
