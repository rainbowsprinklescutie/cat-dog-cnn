import os
import tensorflow as tf
from PIL import Image

base_dataset_dir = 'dataset'

def aggressive_clean(directory):
    count = 0
    deleted = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                try:
                    # Physically read and decode the file like TF does during training
                    img_raw = tf.io.read_file(file_path)
                    tf.io.decode_image(img_raw)
                except Exception as e:
                    print(f"DELETING CORRUPT FILE: {file_path} - Reason: {e}")
                    os.remove(file_path)
                    deleted += 1
                try:
                    with Image.open(file_path) as img:
                        # Re-saving the image effectively 'trims' the garbage bytes
                        img.convert('RGB').save(file_path, "JPEG")
                except Exception as e:
                    print(f"Could not fix {file_path}: {e}")
                count += 1
                if count % 100 == 0:
                    print(f"Checked {count} images...")

    print(f"\nCleanup Finished!")
    print(f"Total checked: {count}")
    print(f"Total deleted: {deleted}")

aggressive_clean(base_dataset_dir)