import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import os
import shutil
from tkinter import filedialog
import base64
import weaviate
import io

# initiate the Weaviate client
client = weaviate.Client("http://localhost:8080")
# Se crea una lista para las imagenes que se subiran a weaviate
imagenesWeaviate = []

##############################################################################################################
# Menú para elegir si buscar una imagen o si clasificar una carpeta de imágenes
print("¿Qué desea hacer?")
print("1. Clasificar una carpeta de imágenes")
print("2. Buscar una imagen con Weaviate")
print("3. Buscar una imagen en carpeta")
print("4. Salir")
opcion = input("Introduce el número de la opción: ")

if opcion == "1":
    # Se carga el modelo y el tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Mostrar información del modelo
    # print("Modelo: ", model)
    print("Dispositivo: ", device)
    print("Modelo cargado.")
    print("----------------------------------------")

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
    labels = input(
        "Introduce las etiquetas de las imágenes separadas por comas: ")
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
        
        # Se convierte la imagen a base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            encoded_string = encoded_string.decode('utf-8')
            imagen = {
                "image": encoded_string,
                "label": predicted_label
            }
            # Se añade la imagen a la lista de imagenes que se subiran a weaviate
            imagenesWeaviate.append(imagen)
        
        # Se mueve la imagen a la carpeta correspondiente
        if not os.path.exists(os.path.join(output_folder, predicted_label)):
            os.makedirs(os.path.join(output_folder, predicted_label))
        shutil.move(image_path, os.path.join(
            output_folder, predicted_label, image_file))
        print(f"{image_file} se ha movido a la carpeta {predicted_label}.")
        print("--------------------------------------------------")

    print("Proceso de clasificación de imágenes finalizado.")
    
    # Se suben las imagenes a weaviate
    for imagen in imagenesWeaviate:
        client.create_object(imagen, "Imagen")
    
    # Se abre la carpeta de imágenes clasificadas
    os.startfile(output_folder)
elif opcion == '2':
    model = SentenceTransformer('clip-ViT-B-32')
    
    # Se pide la frase a buscar
    frase = input("Introduce la frase a buscar: ")
    
    # Se tokeniza la frase
    frase_tokenizada = model.encode(frase, convert_to_tensor=True)
    
    # Se busca en weaviate
    results = client.query(f"{{GetFrase(frase: \"{frase}\"){{frase}}}}")
    
    # Se obtiene la imagen más parecida
    imagen = results["data"]["GetFrase"][0]["frase"]
    
    # Se muestra la imagen
    image = Image.open(io.BytesIO(base64.b64decode(imagen)))
    
elif opcion == '3':
    model = SentenceTransformer('clip-ViT-B-32')

    # Get the path to the image folder
    image_folder = filedialog.askdirectory(
        title="Select the image folder")

    # Get all images in the image folder and its subfolders
    image_files = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                image_files.append(os.path.join(root, file))

    # If there are no images in the folder, exit the program
    if len(image_files) == 0:
        print("No images in the selected folder.")
        quit()

    # Get the phrase to search for
    frase = input("Enter the phrase to search for: ")

    # Tokenize the phrase
    frase_tokenizada = model.encode(frase, convert_to_tensor=True)

    # Create a list to store the results
    resultados = []

    # Iterate over the images
    for image_file in image_files:
        # Get the image path
        image_path = os.path.join(image_folder, image_file)

        # Get the image embedding
        imagen_embedding = model.encode(Image.open(image_path), convert_to_tensor=True)

        # Calculate the similarity between the phrase and the image
        similitud = util.pytorch_cos_sim(frase_tokenizada, imagen_embedding)

        # Add the result to the list
        resultados.append([image_file, similitud.item()])

    # Sort the results from highest to lowest similarity
    resultados.sort(key=lambda x: x[1], reverse=True)

    # Print the most similar image
    print("The most similar image is: ", resultados[0][0])

    # Open the image
    os.startfile(os.path.join(image_folder, resultados[0][0]))
elif opcion == '4':
    #se pide input para salir del programa
    input("Pulsa cualquier tecla para salir...")