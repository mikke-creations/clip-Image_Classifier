import torch
import glob
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import os
import shutil
from tkinter import filedialog
import functions

##############################################################################################################
# Menú para elegir si buscar una imagen o si clasificar una carpeta de imágenes
print("¿Qué desea hacer?")
print("1. Clasificar una carpeta de imágenes")
print("2. Buscar una imagen")
print("3. Salir")
opcion = input("Introduce el número de la opción: ")

if opcion == "1":
    # Se cargan el modelo y el tokenizer
    device, model, preprocess = functions.cargarModeloClasificacion()

    # Se pide la carpeta de imágenes
    image_folder, image_files = functions.obtenerImagenes()

    # Si no existe la carpeta "classified_images", se crea
    output_folder = functions.crearCarpeta(image_folder)

    # Se piden las etiquetas de las imágenes
    labels, text = functions.hacerEtiquetas(device)

    # Se clasifican las imágenes
    functions.clasificacionImagenes(device, model, preprocess, image_files, output_folder, labels, text)

elif opcion == '2':
    # Se carga el modelo
    modelo = functions.cargarModeloConsulta()

    # Se pide la carpeta de imágenes
    carpetaImagenes, archivosImagenes = functions.obtenerImagenes()

    # Se buscan las imágenes
    functions.busquedaDeImagenes(modelo, carpetaImagenes, archivosImagenes)

elif opcion == '3':
    #se pide input para salir del programa
    input("Pulsa cualquier tecla para salir...")