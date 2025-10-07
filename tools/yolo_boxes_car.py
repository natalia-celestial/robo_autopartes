#!/usr/bin/env python3
"""
YOLO Box Annotator - Final Version
- 'a': Auto-detect drinks with YOLO and enter Classification Mode.
- Mouse: Manually draw boxes for objects YOLO missed (only in Normal Mode).
- 's': Save labels and auto-rename the file.
- Instructions are printed in the terminal for a clean view.
"""

import cv2
import glob
from pathlib import Path
from ultralytics import YOLO

# ------------------------- CONFIGURACIÓN -------------------------
TODAS_LAS_CLASES = [
    "autoparts_theft", "mask", "hood", "suspicious_person"
]
YOLO_CLASSES_TO_DETECT = [0, 2] # 0: person, 2: car
YOLO_CONFIDENCE_THRESHOLD = 0.40
PLANTILLA_RENOMBRADO = None #"image_atoparts_theft{:03d}"
INICIO_RENOMBRADO = 1
# -----------------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def unique_stem(target_dir: Path, stem: str, ext: str) -> str:
    candidate = stem
    i = 1
    while (target_dir / f"{candidate}{ext}").exists():
        candidate = f"{stem}_{i}"
        i += 1
    return candidate

def write_labels(label_path, boxes, W, H):
    lines = []
    for (cls, x1, y1, x2, y2) in boxes:
        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2), max(y1, y2)
        w = x_max - x_min; h = y_max - y_min
        if w <= 1 or h <= 1: continue
        cx = (x_min + w / 2.0) / W
        cy = (y_min + h / 2.0) / H
        w_norm = w / W; h_norm = h / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

class Annotator:
    def __init__(self, image_paths, labels_dir, active_classes, yolo_model, rename_template, rename_counter):
        self.image_paths = [Path(p) for p in image_paths]
        self.labels_dir = labels_dir
        ensure_dir(self.labels_dir)
        self.active_classes = active_classes
        self.hotkey_to_class_idx = list(active_classes.keys())
        self.hotkey_idx = 0 # Índice para la clase activa en modo manual
        self.i = 0
        self.boxes = []
        self.unclassified_boxes = []
        self.classification_mode = False
        self.current_unclassified_idx = 0
        self.drawing = False # Flag para el dibujo manual
        self.x1, self.y1, self.x2, self.y2 = 0,0,0,0
        self.window = "Anotador YOLO"
        self.yolo_model = yolo_model
        self.rename_template = rename_template
        self.rename_counter = rename_counter
        
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.window, self.on_mouse)

    def get_current_manual_class_idx(self):
        """Devuelve el índice de la clase seleccionada para dibujo manual."""
        if not self.hotkey_to_class_idx: return 0
        return self.hotkey_to_class_idx[self.hotkey_idx]

    def print_instructions(self):
        print("\n" + "="*50)
        if self.classification_mode:
            box_count = len(self.unclassified_boxes)
            print(f"MODO CLASIFICACIÓN: Caja {self.current_unclassified_idx + 1}/{box_count}")
            print("  - Teclas numéricas: Asignar clase a la caja resaltada.")
            print("  - 'd': Borrar caja. | 'n': Saltar caja. | 'q'/ESC: Salir de este modo.")
        else:
            print(f"MODO NORMAL: Viendo '{self.image_paths[self.i].name}' ({self.i+1}/{len(self.image_paths)})")
            if self.hotkey_to_class_idx:
                active_cls_name = TODAS_LAS_CLASES[self.get_current_manual_class_idx()]
                print(f"  Clase para dibujo manual: '{active_cls_name}' (Usa 0-9 para cambiar)")
            print("  - Ratón: Arrastra para dibujar una caja nueva.")
            print("  - 'a': Auto-detectar bebidas. | 's': Guardar y renombrar.")
            print("  - 'n'/Espacio: Siguiente. | 'p': Anterior. | 'u': Deshacer última caja.")
            print("  - 'q'/ESC: Salir.")
        print("="*50)

    def load_current_image(self):
        img_path = self.image_paths[self.i]
        img = cv2.imread(str(img_path))
        if img is None: raise RuntimeError(f"No se pudo leer la imagen: {img_path}")
        self.boxes = []
        self.unclassified_boxes = []
        self.set_mode(is_classification=False)
        return img

    def on_mouse(self, event, x, y, flags, param):
        # El dibujo manual solo funciona si NO estamos en modo clasificación
        if self.classification_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x1, self.y1, self.x2, self.y2 = x, y, x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x2, self.y2 = x, y
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            # Añadir la caja dibujada a la lista de cajas clasificadas
            box_class = self.get_current_manual_class_idx()
            self.boxes.append([box_class, self.x1, self.y1, self.x2, self.y2])
            print(f"Caja manual añadida con la clase '{TODAS_LAS_CLASES[box_class]}'")

    def set_mode(self, is_classification):
        if self.classification_mode != is_classification:
            self.classification_mode = is_classification
            self.print_instructions()

    def pre_annotate_drinks(self):
        if not self.yolo_model: print("[WARN] Modelo YOLO no cargado."); return
        print(f"[INFO] Detectando bebidas en '{self.image_paths[self.i].name}'...")
        img_path = self.image_paths[self.i]
        results = self.yolo_model(img_path, classes=YOLO_CLASSES_TO_DETECT, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        
        self.unclassified_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                self.unclassified_boxes.append([x1, y1, x2, y2])
        
        if self.unclassified_boxes:
            self.current_unclassified_idx = 0
            self.set_mode(is_classification=True)
        else:
            print("  -> No se encontraron bebidas potenciales.")

    def draw(self, img):
        disp = img.copy()
        
        # Cajas clasificadas (manuales o de IA)
        for (cls_idx, x1, y1, x2, y2) in self.boxes:
            color = color_for_class(cls_idx)
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            label = f"{TODAS_LAS_CLASES[cls_idx]}"
            cv2.putText(disp, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Cajas sin clasificar (de IA)
        for i, (x1, y1, x2, y2) in enumerate(self.unclassified_boxes):
            is_active = (i == self.current_unclassified_idx and self.classification_mode)
            color = (0, 255, 255) if is_active else (255, 255, 255)
            thickness = 3 if is_active else 1
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, thickness)
        
        # Caja que se está dibujando manualmente
        if self.drawing:
            color = color_for_class(self.get_current_manual_class_idx())
            cv2.rectangle(disp, (self.x1, self.y1), (self.x2, self.y2), color, 1)
            
        return disp

    def save_current_and_rename(self, img):
        old_img_path = self.image_paths[self.i]
        
        if self.rename_template:
            new_stem = self.rename_template.format(self.rename_counter)
            new_stem_unique = unique_stem(old_img_path.parent, new_stem, old_img_path.suffix)
            label_stem = new_stem_unique
        else:
            label_stem = old_img_path.stem

        label_path = self.labels_dir / (label_stem + ".txt")
        H, W = img.shape[:2]
        write_labels(label_path, self.boxes, W, H)
        print(f"[Guardado] Etiqueta: {label_path.name} ({len(self.boxes)} cajas)")

        if self.rename_template:
            new_img_path = old_img_path.with_stem(label_stem)
            try:
                old_img_path.rename(new_img_path)
                self.image_paths[self.i] = new_img_path
                print(f"[Renombrado] Imagen: {old_img_path.name} -> {new_img_path.name}")
                self.rename_counter += 1
            except Exception as e:
                print(f"[ERROR] No se pudo renombrar la imagen: {e}")

    def loop(self):
        img = self.load_current_image()
        self.print_instructions()
        while True:
            disp = self.draw(img)
            cv2.imshow(self.window, disp)
            key = cv2.waitKey(20) & 0xFF

            if self.classification_mode:
                if ord('0') <= key <= ord('9'):
                    hotkey = key - ord('0')
                    if hotkey < len(self.hotkey_to_class_idx):
                        class_to_assign = self.hotkey_to_class_idx[hotkey]
                        box_to_classify = self.unclassified_boxes.pop(self.current_unclassified_idx)
                        self.boxes.append([class_to_assign, *box_to_classify])
                        if not self.unclassified_boxes or self.current_unclassified_idx >= len(self.unclassified_boxes):
                            self.set_mode(is_classification=False)
                        else: self.print_instructions()
                elif key == ord('d'):
                    if self.unclassified_boxes:
                        self.unclassified_boxes.pop(self.current_unclassified_idx)
                        if not self.unclassified_boxes or self.current_unclassified_idx >= len(self.unclassified_boxes):
                            self.set_mode(is_classification=False)
                elif key == ord('n'):
                    if self.unclassified_boxes:
                        self.current_unclassified_idx = (self.current_unclassified_idx + 1) % len(self.unclassified_boxes)
                        self.print_instructions()
                elif key in (ord('q'), 27):
                    self.unclassified_boxes = []
                    self.set_mode(is_classification=False)
            else: # Modo Normal
                if key in (ord('q'), 27): break
                elif key in (ord('n'), 32):
                    if self.i < len(self.image_paths) - 1:
                        self.i += 1; img = self.load_current_image()
                    else: print("¡Has llegado a la última imagen!")
                elif key == ord('p'):
                    if self.i > 0:
                        self.i -= 1; img = self.load_current_image()
                elif key == ord('s'): self.save_current_and_rename(img)
                elif key == ord('a'): self.pre_annotate_drinks()
                elif key == ord('u'): # Deshacer última caja (manual o clasificada)
                    if self.boxes: self.boxes.pop(); print("Última caja deshecha.")
                elif ord('0') <= key <= ord('9'): # Cambiar clase para dibujo manual
                    hotkey = key - ord('0')
                    if hotkey < len(self.hotkey_to_class_idx):
                        self.hotkey_idx = hotkey
                        self.print_instructions()
        
        cv2.destroyAllWindows()

def color_for_class(idx):
    palette = [(56, 56, 255), (92, 219, 92), (255, 178, 29), (94, 218, 121)]
    return palette[idx % len(palette)]

def main():
    print("[INFO] Cargando modelo YOLOv8n...")
    try:
        model = YOLO('yolo11n.pt')
        print("[INFO] Modelo YOLO cargado.")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo YOLO: {e}"); model = None

    print("\n--- Configuración de Clases ---")
    print("Clases disponibles para asignar:")
    for i, name in enumerate(TODAS_LAS_CLASES):
        print(f"  {i}: {name}")
    
    active_classes = {}
    while not active_classes:
        try:
            raw_input_str = input("\nIntroduce los NÚMEROS de las clases que usarás (ej: 2, 3): ")
            selected_indices = [int(i.strip()) for i in raw_input_str.split(",")]
            temp_classes = {idx: TODAS_LAS_CLASES[idx] for idx in selected_indices if 0 <= idx < len(TODAS_LAS_CLASES)}
            if temp_classes: active_classes = dict(sorted(temp_classes.items()))
            else: print("No se seleccionó ninguna clase válida.")
        except (ValueError, IndexError):
            print("[Error] Entrada inválida.")

    print("\nClases activas para esta sesión:")
    for i, (original_idx, name) in enumerate(active_classes.items()):
        print(f"  Tecla '{i}' -> Clase [{original_idx}] {name}")

    print("\n--- Selección de Carpetas ---")
    images_dir_path = ""
    while not images_dir_path:
        path_str = input("Arrastra tu CARPETA de imágenes y presiona Enter: ").strip().replace("'", "").replace('"', '')
        if Path(path_str).is_dir(): images_dir_path = Path(path_str)
        else: print("[Error] La ruta no es una carpeta válida.")

    labels_dir_path = ""
    while not labels_dir_path:
        path_str = input("Arrastra la CARPETA para guardar las etiquetas y presiona Enter: ").strip().replace("'", "").replace('"', '')
        if path_str: labels_dir_path = Path(path_str)
        else: print("[Error] La ruta no puede estar vacía.")

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_paths = sorted([p for ex in exts for p in glob.glob(str(images_dir_path / ex))])

    if not image_paths:
        print(f"[Error] No se encontraron imágenes en: {images_dir_path}"); return

    ann = Annotator(image_paths, labels_dir_path, active_classes, model, 
                    rename_template=PLANTILLA_RENOMBRADO, rename_counter=INICIO_RENOMBRADO)
    
    print("\n--- ¡Listo para anotar! ---")
    if ann.rename_template:
        print(f"Renombrado automático activado con plantilla: '{ann.rename_template}' (inicia en {ann.rename_counter})")
    
    cv2.waitKey(1)
    ann.loop()

if __name__ == "__main__":
    main()
