import os
import sharp
import base64
import weaviate

# initiate the Weaviate client
client = weaviate.Client("http://localhost:8080")

# get list of meme files
memeFiles = os.listdir('./memesEjemplo')

# create list of meme objects
memes = []
for memeFile in memeFiles:
with open(f'./memesEjemplo/{memeFile}', 'rb') as f:
imageBuffer = f.read()
b64 = base64.b64encode(imageBuffer).decode('utf-8')
meme = {
'image': b64,
'name': memeFile.split('.')[0].replace('_', ' ')
}
memes.append(meme)

# add memes to Weaviate
for meme in memes:
client.create_object(meme, 'Meme')