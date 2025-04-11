import cv2
import numpy as np
import json
import os
from PIL import Image
import argparse

# Argumente aus der Kommandozeile lesen
parser = argparse.ArgumentParser(description="Erzeuge binäre Masken aus Cityscapes-Labeln")
parser.add_argument('--input_dir', type=str, required=True, help='Pfad zum Cityscapes-Val-Verzeichnis (enthält z. B. frankfurt/...)')
parser.add_argument('--output_dir', type=str, required=True, help='Zielverzeichnis, in dem Bilder und Masken gespeichert werden')
args = parser.parse_args()

base_path = args.input_dir
base_dir = args.input_dir

# Lese alle Städte (Unterordner) im input_dir aus
cities = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# Zielverzeichnisse für Bilder und Masken
image_output_dir = os.path.join(args.output_dir, "images")
mask_output_dir = os.path.join(args.output_dir, "masks")

os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)





# Basispfad zu den Cityscapes-Daten
#base_path = "/work/rn583pgoa-workdir/Cityscapes/val/"
#base_dir="/work/rn583pgoa-workdir/Cityscapes/val/"
#cities = ['frankfurt', 'lindau', 'munster']

# Zielverzeichnisse für Bilder und Masken
#image_output_dir = "/work/rn583pgoa-workdir/val/images"
#mask_output_dir = "/work/rn583pgoa-workdir/val/masks"

# Stelle sicher, dass die Zielverzeichnisse existieren
#os.makedirs(image_output_dir, exist_ok=True)
#os.makedirs(mask_output_dir, exist_ok=True)

# Array, um die Dateinamen zu speichern
pictures = {}

# Durchlaufe die Städte
for city in cities:
    city_dir = os.path.join(base_path, city)
    
    # Erstelle ein Unterarray für die Stadt
    pictures[city] = []
     # Durchlaufe alle Dateien in den Unterordnern der Stadt
    for filename in os.listdir(city_dir):
        # Prüfe auf die gewünschten Endungen
        if filename.endswith("_gtFine_polygons.json"):
            # Extrahiere den String vor "_gtFine_polygons.json"
            base_name = filename.split("_gtFine_polygons.json")[0]
            pictures[city].append(base_name)

# Ausgabe der gespeicherten Dateinamen
#print(pictures)



# Erstelle binäre Masken für jedes Bild im Array
for city in cities:
    for base_name in pictures[city]:
        # Lade das Bild und die zugehörige JSON-Datei
        json_path = os.path.join(base_dir, city, f"{base_name}_gtFine_polygons.json")
        image_path = os.path.join(base_dir, city, 'leftImg8bit', f"{base_name}_leftImg8bit.png")           
        # Lade das Originalbild
        image_og = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_og, cv2.COLOR_BGR2RGB)

        # Erstelle eine leere Maske
        mask = np.zeros((1024, 2048), dtype=np.uint8)
        
        # Lade die Polygondaten aus der JSON-Datei
        with open(json_path, 'r') as f:
            results = json.load(f)

        all_polygons = []
        
        # Iteriere über alle Objekte und extrahiere die Polygone für Autos
        for item in results["objects"]:
            if item["label"] == "car":
                polygon_points = np.array(item['polygon'], np.int32)
                polygon_points = polygon_points.reshape((-1, 1, 2))
                all_polygons.append(polygon_points)

        # Füge alle Polygone zur Maske hinzu
        for polygon in all_polygons:
            cv2.fillPoly(mask, [polygon], 255)

        # Speichere die Maske als binäre PNG-Datei
        mask_save_path = os.path.join(mask_output_dir, f"{base_name}_gtFine_binary.png")
        mask_image = Image.fromarray(mask)
        mask_image.save(mask_save_path)

        # Speichere das Originalbild mit der Maske (nur das Originalbild in RGB)
        image_save_path = os.path.join(image_output_dir, f"{base_name}_leftImg8bit.png")
        cv2.imwrite(image_save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        print(f"Maske gespeichert: {mask_save_path}")
        print(f"Originalbild gespeichert: {image_save_path}") 
