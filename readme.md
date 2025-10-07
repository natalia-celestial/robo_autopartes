# robo_autopartes

Guía rápida (en español) para **instalar**, **anotar videos con YOLO** y **preparar el dataset** siguiendo el mismo flujo de trabajo.

---

## 0) Requisitos

- **GPU NVIDIA** con drivers y CUDA funcionando (recomendado, aunque CPU también sirve pero más lento)
- **Python 3.11**
- **Git**
- Espacio en disco para videos, frames e labels (decenas de miles de imágenes si se extraen muchos fotogramas)

> Consejo: usa VS Code si quieres, pero todo se puede hacer desde terminal.

---

## 1) Clonar el repo e instalar dependencias

```bash
# 1) Clonar
git clone https://github.com/natalia-celestial/robo_autopartes.git
cd robo_autopartes

# 2) Crear entorno (opción A: uv)
pip install uv
uv venv
source ./.venv/bin/activate  # En Windows: .\.venv\Scripts\activate

# 3) Instalar dependencias
uv pip install -r requirements.txt
# Si hiciera falta (según tu sistema):
uv pip install ultralytics opencv-python
```

**Pesos YOLO:** coloca `yolo11n.pt` (o tu checkpoint YOLO) en la raíz del repo. Si lo pones en otra ruta, ajusta `--weights` en los comandos.

---

## 2) Estructura de carpetas

El repo asume esta organización mínima:

```
robo_autopartes/
├─ tools/
│  ├─ yolo_boxes_car.py        # Script principal de auto-etiquetado + revisión manual
│  ├─ extract_frames.py        # (opcional) Para extraer frames de videos
│  └─ yolo_txt_to_coco.py      # (opcional) Convertir YOLO -> COCO
├─ dataset/
│  ├─ videos/                  # Aquí van los videos originales
│  │  ├─ Nat/                  
│  │  └─ Pedro/                
│  ├─ frames/                  # Frames extraídos por persona
│  │  ├─ Nat/
│  │  └─ Pedro/
│  └─ labels/                  # Etiquetas YOLO por persona (txt)
│     ├─ Nat/
│     └─ Pedro/
└─ ...
```

> Esta separación por persona evita choques de nombres y facilita el merge.

---

## 3) Dividir los 70 videos (50/50)

Pon tus 35 videos en `dataset/videos/Nat/` y los otros 35 en `dataset/videos/Pedro/` (puedes cambiar los nombres de las carpetas por los de cada persona).

```bash
mkdir -p dataset/videos/Nat dataset/videos/Pedro
# Copia/pega los archivos .mp4/.mov/.avi correspondientes en cada carpeta
```

---

## 4) Extraer frames de los videos (opcional pero recomendado)

Usa el script de extracción (ajusta `--fps`):

```bash
# Para Nat
mkdir -p dataset/frames/Nat
python tools/extract_frames.py \
  --in_dir dataset/videos/Nat \
  --out_dir dataset/frames/Nat \
  --fps 2 \
  --rename "nat_{:06d}.jpg"

# Para Pedro
mkdir -p dataset/frames/Pedro
python tools/extract_frames.py \
  --in_dir dataset/videos/Pedro \
  --out_dir dataset/frames/Pedro \
  --fps 2 \
  --rename "pedro_{:06d}.jpg"
```

> Si ya tienes imágenes sueltas, puedes saltarte este paso y apuntar el anotador directamente a la carpeta de imágenes.

---

## 5) Anotar con el script YOLO (`yolo_boxes_car.py`)

Este script:
- carga un modelo YOLO preentrenado,
- hace predicción automática de objetos,
- te deja **aceptar/editar** rápidamente y guardar etiquetas en formato **YOLO TXT**.

Ejemplo de uso:

```bash
# Para Nat
mkdir -p dataset/labels/Nat
python tools/yolo_boxes_car.py \
  --images dataset/frames/Nat \
  --labels_out dataset/labels/Nat \
  --weights ./yolo11n.pt \
  --conf 0.25 \
  --classes person,car,truck,motorcycle,bicycle,bus \
  --rename_template "nat_{:06d}.jpg"

# Para Pedro
mkdir -p dataset/labels/Pedro
python tools/yolo_boxes_car.py \
  --images dataset/frames/Pedro \
  --labels_out dataset/labels/Pedro \
  --weights ./yolo11n.pt \
  --conf 0.25 \
  --classes person,car,truck,motorcycle,bicycle,bus \
  --rename_template "pedro_{:06d}.jpg"
```

**Notas importantes**
- Las clases deben coincidir con las clases del modelo (`model.names`).
- El script guarda un `.txt` por imagen con las líneas `class x_center y_center w h` **normalizadas 0..1**.
- Renombrar con un prefijo por persona evita colisiones (`nat_000123.jpg`, `pedro_000123.jpg`).

Controles típicos (pueden variar según tu versión):
- `0..9` seleccionan clase del menú
- Click/drag para dibujar/ajustar cajas
- `Enter` / `S` guardar y pasar a la siguiente
- `A`/`D` o flechas para navegar
- `Esc` salir

---

## 6) Unir anotaciones y (opcional) convertir a COCO

Cuando ambos terminen:

```bash
# 1) Unir labels
mkdir -p dataset/labels/all
rsync -a dataset/labels/Nat/  dataset/labels/all/
rsync -a dataset/labels/Pedro/  dataset/labels/all/

# 2) Unir frames (si necesitas una sola carpeta)
mkdir -p dataset/images/all
rsync -a dataset/frames/Nat/  dataset/images/all/
rsync -a dataset/frames/Pedro/  dataset/images/all/

# 3) YOLO -> COCO (opcional, si vas a entrenar con COCO)
python tools/yolo_txt_to_coco.py \
  --img_dir dataset/images/all \
  --lbl_dir dataset/labels/all \
  --out_json dataset/annotations_all.json
```

> Si tu conversor necesita `--names`, pásalo con el orden de clases correcto.

---

## 7) Entrenamiento (opcional – RF-DETR rápido)

Si quieres entrenar con RF-DETR usando las carpetas `dataset/images/train|valid` y sus `_annotations.coco.json` dentro, un ejemplo mínimo sería:

```bash
python - <<'PY'
from rfdetr import RFDETRBase
m = RFDETRBase()
m.train(
  dataset_dir="dataset/images",   # debe contener train/ y valid/ con sus _annotations.coco.json
  output_dir="runs/rfdetr_autoparts",
  epochs=20,
  batch_size=2,
  grad_accum_steps=4,
  resolution=560,
  device="cuda",
  run_test=False,
  early_stopping=True
)
PY
```

> Este bloque es solo un ejemplo; ajusta según tu máquina (VRAM, etc.).

---

## 8) Buenas prácticas de Git

- **No subas videos, imágenes, ni pesos (.pt/.pth)** al repo. Están ignorados en `.gitignore`.
- Sube **scripts, configs y documentación**.
- Flujo típico:

```bash
git pull
# haz tus cambios en tools/, README, etc.
git add tools/*.py README.md
git commit -m "feat: mejoras en anotador YOLO"
git push
```

---

## 9) Problemas frecuentes

- **No abre la GUI o falta OpenCV**: `pip install opencv-python`
- **No encuentra pesos YOLO**: verifica la ruta `--weights` y que el archivo exista.
- **Clases no coinciden**: revisa `--classes` y que el script esté usando el **id de clase original** del modelo al guardar.
- **Memoria GPU insuficiente**: baja `--conf`, disminuye resolución/fps al extraer frames, o trabaja por lotes de imágenes.

---


