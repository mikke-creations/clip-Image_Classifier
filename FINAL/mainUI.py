import functions
import tkinter as tk

##############################################################################################################

def clasificar_carpeta(labels):
    # Se cargan el modelo y el tokenizer
    device, model, preprocess = functions.cargarModeloClasificacion()
    # Se pide la carpeta de imágenes
    image_folder, image_files, output_folder = functions.obtenerImagenesClasificador()
    # Si no existe la carpeta "classified_images", se crea
    output_folder = functions.crearCarpeta(image_folder)
    # Se piden las etiquetas de las imágenes
    labels, text = functions.hacerEtiquetas(device, labels)
    # Se clasifican las imágenes
    functions.clasificarImagenes(device, model, preprocess, image_folder, image_files, output_folder, labels, text, 0.9)

def buscar_imagen(frase):
    # Se carga el modelo
    modelo = functions.cargarModeloConsulta()
    # Se pide la carpeta de imágenes
    carpetaImagenes, archivosImagenes = functions.obtenerImagenesConsulta()
    # Se buscan las imágenes
    functions.busquedaDeImagen(frase, modelo, carpetaImagenes, archivosImagenes)

def mostrar_menu():
    # Crea una ventana
    ventana = tk.Tk()

    # Define una función para ejecutar cuando se selecciona una opción
    def seleccionar_opcion():
        opcion = opcion_var.get()
        if opcion == 1:
            labels = entry_labels.get()
            clasificar_carpeta(labels)
        elif opcion == 2:
            frase = entry_frase.get()
            buscar_imagen(frase)
        elif opcion == 3:
            ventana.destroy()

    # Crea una variable de control para almacenar la opción seleccionada
    opcion_var = tk.IntVar()

    # Crea los elementos del menú
    tk.Label(ventana, text="¿Qué desea hacer?").pack()
    tk.Frame(ventana, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=10, pady=5)

    opcion_clasificar = tk.Radiobutton(ventana, text="Clasificar una carpeta de imágenes", variable=opcion_var, value=1)
    opcion_clasificar.pack()
    tk.Label(ventana, text="Etiquetas (separadas con comas):").pack()
    entry_labels = tk.Entry(ventana, state=tk.DISABLED)  # Campo de entrada desactivado por defecto
    entry_labels.pack()

    tk.Frame(ventana, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=10, pady=5)

    opcion_buscar = tk.Radiobutton(ventana, text="Buscar una imagen", variable=opcion_var, value=2)
    opcion_buscar.pack()
    tk.Label(ventana, text="Frase a buscar:").pack()
    entry_frase = tk.Entry(ventana, state=tk.DISABLED)  # Campo de entrada desactivado por defecto
    entry_frase.pack()

    tk.Frame(ventana, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=10, pady=5)

    # Funciones para activar/desactivar los campos de entrada según la opción seleccionada
    def activar_entry_labels():
        if opcion_var.get() == 1:
            entry_labels.config(state=tk.NORMAL)
            entry_frase.config(state=tk.DISABLED)
            entry_frase.delete(0, tk.END)
        else:
            entry_labels.delete(0, tk.END)
            entry_labels.config(state=tk.DISABLED)

    def activar_entry_frase():
        if opcion_var.get() == 2:
            entry_frase.config(state=tk.NORMAL)
            entry_labels.config(state=tk.DISABLED)
            entry_labels.delete(0, tk.END)
        else:
            entry_frase.delete(0, tk.END)
            entry_frase.config(state=tk.DISABLED)

    # Asocia las funciones de activación/desactivación con los radiobuttons
    opcion_clasificar.config(command=activar_entry_labels)
    opcion_buscar.config(command=activar_entry_frase)

    # Crea el botón para ejecutar la opción seleccionada
    tk.Button(ventana, text="Ejecutar", command=seleccionar_opcion).pack()

    # Mantiene la ventana abierta
    ventana.mainloop()

if __name__ == "__main__":
    mostrar_menu()
