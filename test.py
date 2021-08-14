import tensorflow as tf
import numpy as np
from matplotlib import gridspec
import matplotlib.pylab as plt
from style_transfer.utils import load_image
from PIL import Image
import warnings
from style_transfer.utils import clip_0_1, tensor_to_image
warnings.filterwarnings("ignore")


def show_n(images, titles=('',)):
    n = len(images)
    image_sizes = [np.array(image).shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    plt.show()


content_image = load_image("./content.jpg", (1024, 1024))
style_image = load_image("./style.jpg", (256, 256))

model = tf.keras.models.load_model("style_transfer/models/arbitrary/")
outputs = model(tf.constant(content_image), tf.constant(style_image))[0][0]
plt.imshow(outputs)
plt.axis("off")
plt.savefig(f"./output_style.jpg")
print("finished")
