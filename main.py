import os
import matplotlib.pyplot as plt
from keras import Sequential, layers
from matplotlib import image as mpimg

from tensorflow.keras.utils import image_dataset_from_directory

base_dataset_dir = 'dataset'

# Visualize the Data

cat_dir = os.path.join(base_dataset_dir, 'cats_set')
dog_dir = os.path.join(base_dataset_dir, 'dogs_set')

cat_names = os.listdir(cat_dir)
dog_names = os.listdir(dog_dir)

cat_images = [os.path.join(cat_dir, cat_name) for cat_name in cat_names]
dog_images = [os.path.join(dog_dir, dog_name) for dog_name in dog_names]

for i, img_path in enumerate(cat_images[:8] + dog_images[:8]):
    splot = plt.subplot(4, 4, i+1)
    splot.axis('off')
    print(img_path)
    img = mpimg.imread(img_path)
    plt.imshow(img)
plt.show()

# Splitting Dataset

train_datagen = image_dataset_from_directory(base_dataset_dir,
                                             image_size=(200, 200),
                                             subset='training',
                                             seed=1,
                                             validation_split=0.1,
                                             batch_size=32)
test_datagen = image_dataset_from_directory(base_dataset_dir,
                                            image_size=(200, 200),
                                            subset='validation',
                                            seed=1,
                                            validation_split=0.1,
                                            batch_size=32)

class_names = train_datagen.class_names
print(f"class_names: {class_names}")

for images, labels in train_datagen.take(1):
    batch_actual_size = images.shape[0]
    for index in range(batch_actual_size):
        ax = plt.subplot(4, 8, index+1)
        plt.imshow(images[index].numpy().astype("uint8"))
        plt.title(class_names[labels[index]])
        plt.axis('off')
plt.tight_layout()
plt.show()

# Model Architecture

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.summary()
