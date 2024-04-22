import os
import PIL.Image as pillow
import base64
import weaviate
from sentence_transformers import SentenceTransformer, util
import io
##############################################################################################################

# initiate the Weaviate client
client = weaviate.Client("http://localhost:8080")

# Se crea el modelo de búsqueda
model = SentenceTransformer('clip-ViT-B-32')

# Se pide la frase a buscar
frase = input("Introduce la frase a buscar: ")

# Se obtienen los objetos de la clase Imagen
imagenes = client.objects.get(class_name='Imagen')

# Se tokeniza el vector de la frase
frase_vector = model.encode(frase)

# Se crea una lista para guardar los resultados
resultados = []

# Se recorren las imágenes siendo codificadas por base64
for imagen in imagenes:
    imagen_base64 = imagen['image']
    # Se decodifica la imagen
    imagen_decodificada = base64.b64decode(imagen_base64)
    # Se abre la imagen
    imagen_abierta = pillow.open(io.BytesIO(imagen_decodificada))
    # Se codifica la imagen
    imagen_codificada = base64.b64encode(imagen_abierta.tobytes()).decode('utf-8')
    # Se obtiene el vector de la imagen
    imagen_vector = model.encode(imagen_codificada)
    # Se calcula la similitud entre la frase y la imagen
    similitud = util.pytorch_cos_sim(frase_vector, imagen_vector)
    # Se añade el resultado a la lista de resultados
    resultados.append((imagen['name'], similitud))

# Se ordenan los resultados
resultados.sort(key=lambda x: x[1], reverse=True)

# Se muestra el resultado más parecido
print(f"La imagen más parecida a la frase '{frase}' es: {resultados[0][0]}")

# Se muestra la imagen
image = pillow.open(io.BytesIO(base64.b64decode(imagenes[0]['image'])))

# Se muestra la imagen
image.show()

# Se cierra el cliente
client.close()

# Se cierra el programa con input
input("Pulse cualquier tecla para salir...")