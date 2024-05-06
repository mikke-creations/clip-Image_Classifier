import functions

run = True

while run:
    print("What would you like to do?")
    print("1. Sort a folder of images")
    print("2. Search for an image")
    print("3. Exit")
    option = input("Enter the option number: ")

    if option == "1":
        # Model and tokenizer are loaded
        device, model, preprocess = functions.loadSortingModel()

        # Images folder is requested
        image_folder, image_files, output_folder = functions.getImagesSorter()

        # If "sorted_images" folder does not exist, it is created
        output_folder = functions.createFolder(image_folder)

        # Image tags are requested
        labels = input(
            "Enter tags separated by commas: ")
        
        # Image tags are requested
        labels, text = functions.makeLabels(device, labels)

        # Images are classified
        functions.sortImages(device, model, preprocess, image_folder, image_files, output_folder, labels, text, 0.9)

    elif option == '2':
        # Mdel is loaded
        model = functions.loadQueryModel()

        # Images folder is requested
        imagesFolder, imagesFiles = functions.getQueryImages()

        # The phrase to search is requested
        phrase = input("Enter a phrase to search: ")

        # Images are searched
        functions.imageSearch(phrase, model, imagesFolder, imagesFiles)

    elif option == '3':
        input("Press any key to exit...")
        run = False
    