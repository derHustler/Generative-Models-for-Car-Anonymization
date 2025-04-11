# Bachelorarbeit: Vergleich von GANs und Diffusionsmodellen zur Autogenerierung in Cityscapes

Dieses Repository enthält den Code und Anleitungen zur Reproduktion meiner Bachelorarbeit, in der verschiedene Bildgenerierungsmodelle (DP_GAN, OASIS, Kolors, SDXL) miteinander verglichen werden.

## Inhalt

1. [Setup](#setup)  
2. [Cityscapes vorbereiten](#cityscapes-vorbereiten)  
3. [Masken generieren](#masken-generieren)  

# Setup

### 1. Kolors einrichten

Um das Kolors-Repository für Inpainting zu klonen und die entsprechende Umgebung zu erstellen, folge bitte den Anweisungen hier:  
https://github.com/Kwai-Kolors/Kolors/blob/master/inpainting/README.md

### 2. DP_GAN einrichten

Um das DP_GAN-Repository zu klonen und die entsprechende Umgebung zu erstellen, folge bitte den Anweisungen hier:  
https://github.com/sj-li/DP_GAN/tree/main

Die Checkpoints der vortrainierten Modelle sind hier als ZIP-Dateien verfügbar. Kopiere sie in den Ordner `checkpoints` (Standard ist `./checkpoints`, falls nicht vorhanden bitte erstellen) und entpacke sie dort. Die Ordnerstruktur sollte wie folgt aussehen:

```bash
.
├── DP_GAN/
│   └── scripts/
│   └── checkpoints/
│       └── dp_gan_cityscapes/
│   └── [...]
└── [...]
```

### 3. OASIS einrichten

Um das OASIS-Repository zu klonen, das vortrainierte Cityscapes-Modell herunterzuladen und die Umgebung zu erstellen, folge bitte den Anweisungen hier:  
https://github.com/boschresearch/OASIS

### 4. SDXL einrichten

```bash
conda create --name sdxl 
conda activate sdxl
pip install -r requirementsSDXL.txt
```

---

## Cityscapes vorbereiten

Für die Vorbereitung des Cityscapes-Datensatzes folge bitte den Anweisungen aus folgendem Repository:  
https://github.com/NVlabs/SPADE

## Masken generieren

Nutze das Skript `mask_generator.py`, um binäre Masken aus den Cityscapes-Labelmaps zu erzeugen und gemeinsam mit den Bildern in einem Ordner abzuspeichern. Ersetze `--input_dir` durch das zu bearbeitende Set und `--output_dir` durch das gewünschte Ausgabeverzeichnis.

```bash
conda activate sdxl
python mask_generator.py --input_dir ./datasets/cityscapes/val --output_dir ./val
```
