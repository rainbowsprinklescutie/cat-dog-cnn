import matplotlib.pyplot as plt
import pandas as pd
from keras import Sequential, layers

from tensorflow.keras.utils import image_dataset_from_directory
from init import base_dataset_dir

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
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# Model Compilation and Training

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(train_datagen,
          epochs=1,
          validation_data=test_datagen)

model.save('model.tensorflow.h5')

# Model Evaluation

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
