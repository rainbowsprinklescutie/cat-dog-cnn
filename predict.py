import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


def predict_image(image_path, model):
    """Predicts if an image is a Dog or a Cat using the provided model."""
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        print("Started predicting image...")
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(200, 200))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        result = model.predict(img_array)
        # Accessing [0][0] assuming a single output neuron with Sigmoid activation
        prediction = "Dog" if result[0][0] >= 0.5 else "Cat"
        print(f"Result for {image_path}: {prediction} (Confidence: {result[0][0]:.2f})")

    except Exception as e:
        print(f"[!] ERROR processing image {image_path}: {e}")


if __name__ == "__main__":
    # --- Configuration & Model Loading ---
    # Updated to your actual filename: model.tensorflow.h5
    model_path = "model.tensorflow.h5"

    # If an argument is provided via CLI, use it to override the default
    if len(sys.argv) == 2:
        model_path = sys.argv[1]

    # Ensure we are looking in the script's directory if no path is specified
    if not os.path.isabs(model_path) and not model_path.startswith("./"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_path)

    try:
        print(f"[*] Current Time: 10:00 AM | Date: June 13, 2026")
        print(f"[*] Loading model from: {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("[+] Model loaded successfully.")
    except Exception as e:
        print(f"[!] CRITICAL ERROR: Could not load model. \nDetails: {e}")
        sys.exit(1)

    # --- Run Predictions ---
    test_images = [
        'dataset/cats_set/320.jpg',
        'dataset/dogs_set/5510.jpg'
    ]

    for path in test_images:
        predict_image(path, model)