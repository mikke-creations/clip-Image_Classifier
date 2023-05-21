import os
import PIL.Image as pillow
import base64
import weaviate

# initiate the Weaviate client
client = weaviate.Client("http://localhost:8080")

client.batch.configure(
  batch_size=100,
  dynamic=False,
  timeout_retries=3,
  callback=weaviate.util.check_batch_result,
  consistency_level=weaviate.data.replication.ConsistencyLevel.ALL,
)

# get list of image files
image_files = os.listdir('./unclassified_images')

with client.batch as batch:
    # create list of image objects
    images = []
    for image_file in image_files:
        with open(f'./unclassified_images/{image_file}', 'rb') as f:
            image = pillow.open(f)
            b64 = base64.b64encode(image.tobytes()).decode('utf-8')
            image_obj = {
                'image': b64,
                'text': image_file
            }
            
            # Se muestra el image_obj
            # print(image_obj)
            
            batch.add_data_object(image_obj, "Imagen")

    try:
        batch.create_objects()
    except Exception as e:
        print(f"Error: {e}")