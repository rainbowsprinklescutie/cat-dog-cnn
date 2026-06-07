import sys

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def predict_image(image_path):
    model = tf.keras.models.load_model(sys.argv[1])
    img = image.load_img(image_path, target_size=(200, 200))
    plt.imshow(img)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)
    print("Dog" if result >= 0.5 else "Cat")

if len(sys.argv) != 2:
    print("\n[!] ERROR: Invalid number of arguments.")
    print(f"Current Time: 11:13 PM | Date: June 7, 2026")
    print("-" * 50)
    print("USAGE: python predict.py <model_path>")
    print("EXAMPLE: python predict.py model.tensorflow.h5 cat.jpg")
    print("-" * 50)
    sys.exit(1)

predict_image('dataset/cats_set/320.jpg')
predict_image('dataset/dogs_set/5510.jpg')
