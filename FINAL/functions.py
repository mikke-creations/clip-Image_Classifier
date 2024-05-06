import torch
import glob
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import os
import shutil
from tkinter import filedialog

def loadSortingModel():
    # Model and tokenizer are loaded
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Show model information
    print("Device: ", device)
    print("Model is loaded.")
    print("----------------------------------------")

    return device, model, preprocess

def getImagesSorter():
    # Images folder is requested
    image_folder = filedialog.askdirectory(
        title="Select the images folder")
    print("Selected folder: ", image_folder)

    # Images are obtained from the folder
    image_files = [
        f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))
    ]

    # If there are no images in the folder, the program ends
    if len(image_files) == 0:
        print("There are no images in the selected folder.")
        quit()

    # If the "sorted_images" folder does not exist, it is created
    if not os.path.exists(os.path.join(image_folder, "sorted_images")):
        os.makedirs(os.path.join(image_folder, "sorted_images"))
        print("The folder 'sorted_images' has been created.")

    output_folder = os.path.join(image_folder, "sorted_images")
    
    return image_folder, image_files, output_folder

def createFolder(image_folder):
    # If the "sorted_images" folder does not exist, it is created
    if not os.path.exists(os.path.join(image_folder, "sorted_images")):
        os.makedirs(os.path.join(image_folder, "sorted_images"))
        print("The folder 'sorted_images' has been created.")

    output_folder = os.path.join(image_folder, "sorted_images")

    return output_folder

def makeLabels(device,labels):
    # Image tags are requested
    labels = labels.split(",")
    labels = [label.strip() for label in labels]

    # Empty tags are removed
    labels = list(filter(None, labels))

    # If there are no tags, the program ends
    if len(labels) == 0:
        print("No tags have been entered.")
        quit()

    # Repeated tags are removed
    labels = list(dict.fromkeys(labels))

    # The entered tags are displayed
    print("Entered tags: ", labels)

    # The labels obtained are tokenized
    text = clip.tokenize(labels).to(device)

    return labels, text

def sortImages(device, model, preprocess, image_folder, image_files, output_folder, labels, text, rate):
    print("Image sorting process has started...")

    for image_file in image_files:
        # The image is obtained
        image_path = os.path.join(image_folder, image_file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # The characteristics of the image and text are obtained
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # If the probability value is less than 0.9, it is sorted as "other"
        if probs[0, probs.argmax()] < rate:
            predicted_label = "other"
        else:
            predicted_label = labels[probs.argmax()]

        # Labels and their probabilities are shown
        print(f"Labels: {labels}")
        print(f"Probabilities: {probs}")

        # The predicted label is displayed in the console
        print(f"Predicted label: {predicted_label}")

        # The image is moved to the corresponding folder
        if not os.path.exists(os.path.join(output_folder, predicted_label)):
            os.makedirs(os.path.join(output_folder, predicted_label))
        shutil.move(image_path, os.path.join(output_folder, predicted_label, image_file))
        print(f"{image_file} has been moved from {image_path} to folder {predicted_label}.")
        print("--------------------------------------------------")

    print("Image sorting process has finalized.")
    # The sorting images folder opens
    os.startfile(output_folder)

##############################################################################################################

def loadQueryModel():
    # Model is loaded
    model = SentenceTransformer('clip-ViT-B-32')
    return model

def getQueryImages():
    # Select the images folder
    image_folder = filedialog.askdirectory(
        title="Select the images folder")

    # Images are obtained from the folder and its subfolders
    image_files = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                image_files.append(os.path.join(root, file))

    # If there are no images in the folder, the program closes
    if len(image_files) == 0:
        print("No images have been found in the given directory.")
        quit()

    return image_folder, image_files

def imageSearch(phrase, model, image_folder, image_files):
    # The phrase is tokenized
    tokenized_phrase = model.encode(phrase, convert_to_tensor=True)

    # A list is initialized to save the results
    results = []

    # We iterate over the images
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        imagen_embedding = model.encode(Image.open(image_path), convert_to_tensor=True)

        # We calculate the similarity between the phrase and the image
        similitud = util.pytorch_cos_sim(tokenized_phrase, imagen_embedding)

        # The result is added to the list
        results.append([image_file, similitud.item()])

    # The results are ordered from greatest to least similarity.
    results.sort(key=lambda x: x[1], reverse=True)

    # We print the most similar image
    print("Most similar image is: ", results[0][0])

    # We open the image
    os.startfile(os.path.join(image_folder, results[0][0]))