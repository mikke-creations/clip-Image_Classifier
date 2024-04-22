import functions

##############################################################################################################

correr = True

while correr:
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
        image_folder, image_files, output_folder = functions.obtenerImagenesClasificador()

        # Si no existe la carpeta "classified_images", se crea
        output_folder = functions.crearCarpeta(image_folder)

        # Se piden las etiquetas de las imágenes
        labels = input(
            "Introduce las etiquetas de las imágenes separadas por comas: ")
        
        # Se piden las etiquetas de las imágenes
        labels, text = functions.hacerEtiquetas(device, labels)

        # Se clasifican las imágenes
        functions.clasificarImagenes(device, model, preprocess, image_folder, image_files, output_folder, labels, text, 0.9)

    elif opcion == '2':
        # Se carga el modelo
        modelo = functions.cargarModeloConsulta()

        # Se pide la carpeta de imágenes
        carpetaImagenes, archivosImagenes = functions.obtenerImagenesConsulta()

        # Se pide la frase a buscar
        frase = input("Introduce la frase a buscar: ")

        # Se buscan las imágenes
        functions.busquedaDeImagen(frase, modelo, carpetaImagenes, archivosImagenes)

    elif opcion == '3':
        #se pide input para salir del programa
        input("Pulsa cualquier tecla para salir...")
        correr = False
    