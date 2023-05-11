import torch
import clip
from PIL import Image
import os
import shutil
from tkinter import filedialog

##############################################################################################################

# Se carga el modelo y el tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

##############################################################################################################

# Se pide la carpeta de imágenes
image_folder = filedialog.askdirectory(
    title="Selecciona la carpeta de imágenes")
print("Carpeta seleccionada: ", image_folder)

# Se obtienen las imágenes de la carpeta
image_files = [
    f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))
]

# Si no hay imágenes en la carpeta, se termina el programa
if len(image_files) == 0:
    print("No hay imágenes en la carpeta seleccionada.")
    quit()

# Si no existe la carpeta "classified_images", se crea
if not os.path.exists(os.path.join(image_folder, "classified_images")):
    os.makedirs(os.path.join(image_folder, "classified_images"))
    print("Se ha creado la carpeta 'classified_images'.")

output_folder = os.path.join(image_folder, "classified_images")

##############################################################################################################

# Se piden las etiquetas de las imágenes
labels = input("Introduce las etiquetas de las imágenes separadas por comas: ")
labels = labels.split(",")
labels = [label.strip() for label in labels]

# Se eliminan las etiquetas vacías
labels = list(filter(None, labels))

# Si no hay etiquetas, se termina el programa
if len(labels) == 0:
    print("No se han introducido etiquetas.")
    quit()

# Se eliminan las etiquetas repetidas
labels = list(dict.fromkeys(labels))

# Se muestran las etiquetas introducidas
print("Etiquetas introducidas: ", labels)

# Se piden confirmación para continuar
confirm = input("¿Son correctas las etiquetas? (s/n): ")
if confirm.lower() != "s":
    quit()

# Se tokenizan las etiquetas obtenidas
text = clip.tokenize(labels).to(device)

##############################################################################################################

print("Comienza el proceso de clasificación de imágenes...")

for image_file in image_files:
    # Se obtiene la imagen
    image_path = os.path.join(image_folder, image_file)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Se obtienen las características de la imagen y del texto
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Si el valor de probabilidad es menor a 0.9, se clasifica como "otro"
    if probs[0, probs.argmax()] < 0.90:
        predicted_label = "other"
    else:
        predicted_label = labels[probs.argmax()]

    # Se muestran las etiquetas y sus probabilidades
    print(f"Etiquetas: {labels}")
    print(f"Probabilidades: {probs}")

    # Laetiqueta predicha se muestra en la consola
    print(f"Etiqueta predicha: {predicted_label}")

    # Se mueve la imagen a la carpeta correspondiente
    if not os.path.exists(os.path.join(output_folder, predicted_label)):
        os.makedirs(os.path.join(output_folder, predicted_label))
    shutil.move(image_path, os.path.join(
        output_folder, predicted_label, image_file))
    print(f"{image_file} se ha movido a la carpeta {predicted_label}.")
    print("--------------------------------------------------")

print("Proceso de clasificación de imágenes finalizado.")
# Se abre la carpeta de imágenes clasificadas
os.startfile(output_folder)
