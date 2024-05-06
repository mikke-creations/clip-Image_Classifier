import functions
import tkinter as tk

def sort_folder(labels):
    # Model and tokenizer are loaded
    device, model, preprocess = functions.loadSortingModel()
    # Images folder is requested
    image_folder, image_files, output_folder = functions.getImagesSorter()
    # If "classified_images" folder does not exist, it's created
    output_folder = functions.createFolder(image_folder)
    # Image tags are requested
    labels, text = functions.makeLabels(device, labels)
    # Images are sorted
    functions.sortImages(device, model, preprocess, image_folder, image_files, output_folder, labels, text, 0.9)

def search_image(phrase):
    # The model is loaded
    model = functions.loadQueryModel()
    # The images folder is requested
    imagesFolder, imageFiles = functions.getQueryImages()
    # Images are searched
    functions.imageSearch(phrase, model, imagesFolder, imageFiles)

def show_menu():
    # Create a window
    window = tk.Tk()

    # Execute an option when it's selected
    def select_option():
        option = option_var.get()
        if option == 1:
            labels = entry_labels.get()
            sort_folder(labels)
        elif option == 2:
            phrase = entry_phrase.get()
            search_image(phrase)
        elif option == 3:
            window.destroy()

    # Create a control variable to store the selected option
    option_var = tk.IntVar()

    # Create menu items
    tk.Label(window, text="What would you like to do?").pack()
    tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=10, pady=5)

    option_sort = tk.Radiobutton(window, text="Sort a folder of images", variable=option_var, value=1)
    option_sort.pack()
    tk.Label(window, text="Tags (separated with commas):").pack()
    entry_labels = tk.Entry(window, state=tk.DISABLED)
    entry_labels.pack()

    tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=10, pady=5)

    option_search = tk.Radiobutton(window, text="Search for an image", variable=option_var, value=2)
    option_search.pack()
    tk.Label(window, text="Phrase to search:").pack()
    entry_phrase = tk.Entry(window, state=tk.DISABLED)
    entry_phrase.pack()

    tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=10, pady=5)

    # Functions to activate/deactivate input fields according to the selected option
    def activate_entry_labels():
        if option_var.get() == 1:
            entry_labels.config(state=tk.NORMAL)
            entry_phrase.config(state=tk.DISABLED)
            entry_phrase.delete(0, tk.END)
        else:
            entry_labels.delete(0, tk.END)
            entry_labels.config(state=tk.DISABLED)

    def activate_entry_phrase():
        if option_var.get() == 2:
            entry_phrase.config(state=tk.NORMAL)
            entry_labels.config(state=tk.DISABLED)
            entry_labels.delete(0, tk.END)
        else:
            entry_phrase.delete(0, tk.END)
            entry_phrase.config(state=tk.DISABLED)

    # Associate the activation/deactivation functions with radiobuttons
    option_sort.config(command=activate_entry_labels)
    option_search.config(command=activate_entry_phrase)

    # Button to execute the selected option
    tk.Button(window, text="Run", command=select_option).pack()

    window.mainloop()

if __name__ == "__main__":
    show_menu()
