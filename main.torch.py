import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

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

train_loader = DataLoader(train_datagen, batch_size=32, shuffle=True)
test_loader = DataLoader(test_datagen, batch_size=32, shuffle=False)

# model = Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Keras: layers.Conv2D(32, (3, 3), input_shape=(200, 200, 3))
        self.conv1 = nn.Conv2d(3, 32, 3)

        # Keras: layers.MaxPooling2D((2, 2)) (We use one object for all pools)
        self.pool = nn.MaxPool2d(2, 2)

        # Keras: layers.Conv2D(64, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, 3)

        # Keras: layers.Conv2D(128, (3, 3))
        self.conv3 = nn.Conv2d(64, 128, 3)

        # Keras: layers.Flatten() -> maps to 128 * 23 * 23 = 67712
        # Keras: layers.Dense(512)
        self.fc1 = nn.Linear(128 * 23 * 23, 512)

        # Keras: layers.Dense(1)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Layer 1: Conv -> Relu -> Pool
        x = self.pool(F.relu(self.conv1(x)))

        # Layer 2: Conv -> Relu -> Pool
        x = self.pool(F.relu(self.conv2(x)))

        # Layer 3: Conv -> Relu -> Pool
        x = self.pool(F.relu(self.conv3(x)))

        # Layer 4: Flatten
        x = torch.flatten(x, 1)

        # Layer 5: Dense(512) -> Relu
        x = F.relu(self.fc1(x))

        # Layer 6: Dense(1) -> Sigmoid
        x = torch.sigmoid(self.fc2(x))

        return x

# Setup Model, loss, and optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# class_names = train_datagen.class_names
# print(f"class_names: {class_names}")
#
# for images, labels in train_datagen.take(1):
#     batch_actual_size = images.shape[0]
#     for index in range(batch_actual_size):
#         ax = plt.subplot(4, 8, index+1)
#         plt.imshow(images[index].numpy().astype("uint8"))
#         plt.title(class_names[labels[index]])
#         plt.axis('off')
# plt.tight_layout()
# plt.show()
#
# # Model Architecture
#

#
# model.summary()
#
# # Model Compilation and Training
#
# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )
#
#
# history = model.fit(train_datagen,
#           epochs=3,
#           validation_data=test_datagen)
#
# model.save('model.tensorflow.h5')
#
# # Model Evaluation
#
# history_df = pd.DataFrame(history.history)
# history_df.loc[:, ['loss', 'val_loss']].plot()
#
# history_df = pd.DataFrame(history.history)
# history_df.loc[:, ['loss', 'val_loss']].plot()
