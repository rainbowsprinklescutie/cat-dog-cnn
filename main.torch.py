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

train_loader = train_datagen # DataLoader(train_datagen, batch_size=32, shuffle=True)
test_loader = test_datagen # DataLoader(test_datagen, batch_size=32, shuffle=False)

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

epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for i , (images_tf, labels_tf) in enumerate(train_loader):
        # 1. Convert TensorFlow Tensor -> NumPy -> PyTorch Tensor
        images = torch.from_numpy(images_tf.numpy())
        labels = torch.from_numpy(labels_tf.numpy())

        # 2. Rescale (0-255 to 0-1) and Reorder (H,W,C to C,H,W)
        # We divide by 255.0 because NNs perform much better with small values
        images = (images / 255.0).permute(0, 3, 1, 2).to(device)
        labels = labels.float().to(device).view(-1, 1)

        optimizer.zero_grad()

        # 3. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 4. Backward pass (calculate gradients)
        loss.backward()

        # 5. Optimize (update weights)
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0: print(f"{i} images done")

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
print("Finished Training")
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
