# Bachelorarbeit · Vergleich von GANs und Diffusions‑Modellen zur Autogenerierung in Cityscapes

Dieses Repository enthält den begleitenden Code zu meiner Bachelorarbeit, in der ich verschiedene Bildgenerierungsmodelle — **DP‑GAN, OASIS, Kolors** und **Stable Diffusion XL (SDXL)** — auf dem Cityscapes‑Datensatz vergleiche.

---
## Inhaltsverzeichnis
1. [Setup](#setup)
   1. [Kolors](#kolors)
   2. [DP‑GAN](#dp_gan)
   3. [OASIS](#oasis)
   4. [SDXL](#sdxl)
2. [Cityscapes vorbereiten](#cityscapes-vorbereiten)
3. [Masken generieren](#masken-generieren)
4. [Bilder generieren](#bilder-generieren)
   1. [Kolors – Steps & Whole](#kolors)
   2. [SDXL – Steps & Whole](#sdxl-1)
   3. [DP‑GAN](#dp_gan-1)
   4. [OASIS](#oasis-1)

---
## Setup<a name="setup"></a>

### Kolors<a name="kolors"></a>
Folge der offiziellen Anleitung, um das **Kolors‑Inpainting**‑Repository zu klonen und die passende Conda‑Umgebung anzulegen:
<https://github.com/Kwai-Kolors/Kolors/blob/master/inpainting/README.md>

Alternativ kannst du das Modell direkt von Hugging Face laden: <https://huggingface.co/Kwai-Kolors/Kolors-Inpainting>

---
### DP‑GAN<a name="dp_gan"></a>
1. Repository klonen: <https://github.com/sj-li/DP_GAN>
2. Checkpoints herunterladen (ZIP) und in **`./checkpoints/`** entpacken. Die Struktur sollte danach so aussehen:

```bash
DP_GAN/
├── checkpoints/
│   └── dp_gan_cityscapes/
└── scripts/
```

---
### OASIS<a name="oasis"></a>
* Repository & Cityscapes‑Checkpoint: <https://github.com/boschresearch/OASIS>

---
### SDXL<a name="sdxl"></a>
```bash
conda create -n sdxl python=3.10
conda activate sdxl
pip install -r requirementsSDXL.txt
```

---
## Cityscapes vorbereiten<a name="cityscapes-vorbereiten"></a>
Zur Konvertierung und Aufteilung des Cityscapes‑Datensatzes nutze die Anweisungen aus SPADE:
<https://github.com/NVlabs/SPADE>

---
## Masken generieren<a name="masken-generieren"></a>
Mit **`mask_generator.py`** erstellst du binäre Masken (Klasse *car*) und kopierst die zugehörigen RGB‑Bilder.

```bash
conda activate sdxl
python mask_generator.py \
  --input_dir  /datasets/cityscapes/val \
  --output_dir ./outputs/val_masks
```

Dies legt folgende Ordner an:

```
outputs/val_masks/
├── images/<city>/*_leftImg8bit.png
└── masks/<city>/*_gtFine_binary.png
```

---
## Bilder generieren<a name="bilder-generieren"></a>
### Kolors<a name="kolors"></a>
* **Steps** (Inpainting jedes Auto einzeln)

```bash
conda activate Kolors
python kolors_steps.py \
  --input_dir  ./outputs/val_masks \
  --output_dir ./outputs/kolors_steps
```

* **Whole** (Gesamtmaske in einem Schritt)

```bash
conda activate Kolors
python kolors_whole.py \
  --input_dir  ./outputs/val_masks \
  --output_dir ./outputs/kolors_whole
```

---
### SDXL<a name="sdxl-1"></a>
* **Steps**
```bash
conda activate sdxl
python sdxl_steps.py \
  --input_dir  ./outputs/val_masks \
  --output_dir ./outputs/sdxl_steps
```

* **Whole**
```bash
conda activate sdxl
python sdxl_whole.py \
  --input_dir  ./outputs/val_masks \
  --output_dir ./outputs/sdxl_whole
```

---
### DP‑GAN<a name="dp_gan-1"></a>
1. Folge der Originaldoku, um Bilder zu erzeugen: <https://github.com/sj-li/DP_GAN>
2. Merge mit Originalbildern & Masken:

```bash
python combineGANpics.py \
  --ganpics      /pfad/zu/dpgan_results \
  --originalpics ./outputs/val_masks/images \
  --masks        ./outputs/val_masks/masks \
  --output       ./outputs/dpgan_merged
```

---
### OASIS<a name="oasis-1"></a>
1. Folge der Originaldoku, um Bilder zu erzeugen: <https://github.com/boschresearch/OASIS>
2. Merge‑Schritt analog zu DP‑GAN:

```bash
python combineGANpics.py \
  --ganpics      /pfad/zu/oasis_results \
  --originalpics ./outputs/val_masks/images \
  --masks        ./outputs/val_masks/masks \
  --output       ./outputs/oasis_merged
```

---
## Kontakt
Fragen oder Probleme? → **ad41liqo@studserv.uni-leipzig.de**

