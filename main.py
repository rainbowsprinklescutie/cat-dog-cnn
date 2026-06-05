import os
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

cat_dir = os.path.join('dataset/cats_set')
dog_dir = os.path.join('dataset/dogs_set')

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
