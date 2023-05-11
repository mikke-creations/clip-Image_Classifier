# curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml?generative_cohere=false&generative_palm=false&image_neural_model=pytorch-resnet50&media_type=image&modules=modules&ref2vec_centroid=false&runtime=docker-compose&weaviate_version=v1.19.2"

import weaviate

client = weaviate.Client("http://localhost:8080")

class_img = {
    "class": "Imagen",
    'vectorizer': 'img2vec-neural',
    'vectorIndexType': 'hnsw',
    'moduleConfig': {
        'img2vec-neural': {
             'imageFields': [
                 'image'
             ]
        }
    },
    "properties": [
        {
            'name': 'image',
            'dataType': ['blob']
        }
    ]
}

#Se crea la clase Imagen y se le asigna el vectorizador img2vec-neural
new_class = client.schema.create_class(class_img)


