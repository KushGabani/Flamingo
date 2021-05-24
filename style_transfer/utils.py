from app import style_transfer
import matplotlib.pylab as plt
import matplotlib
import tensorflow as tf
import numpy as np
from PIL import Image


def tensor_to_image(tensor):
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def crop_center(image):
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image


def load_image(image_path, image_size=(256, 256), preserve_aspect_ratio=True, is_style_image=False):
    img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)


def vgg_layers(layer_names):
    vgg = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(layer).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(feature_map, normalize=True):
    gram = tf.linalg.einsum('bijc,bijd->bcd', feature_map, feature_map)
    if normalize:
        input_shape = tf.shape(feature_map)
        gram /= tf.cast(input_shape[1] * input_shape[2], tf.float32)

    return gram


def style_loss(style_outputs, style_target):
    loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_target[name]) ** 2)
                    for name in style_outputs.keys()])
    return loss


def content_loss(content_outputs, content_target):
    loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_target[name]) ** 2)
                    for name in content_outputs.keys()])
    return loss


def total_variation_loss(img):
    x = img[:, :, 1:, :] - img[:, :, :-1, :]
    y = img[:, 1:, :, :] - img[:, :-1, :, :]

    return tf.reduce_mean(tf.square(x)) + tf.reduce_mean(tf.square(y))


def get_constants():
    return {
        "CONTENT_WEIGHT": 6e0,
        "STYLE_WEIGHT": 2e-3,
        "TV_WEIGHT ": 6e2,
        "LEARNING_RATE ": 1e-3,
        "NUM_EPOCHS ": 2,
        "BATCH_SIZE ": 2,
    }


def save_result(image):
    matplotlib.use('Agg')
    plt.imshow(image)
    plt.axis("off")
    plt.savefig("static/style_transfer/output.png")
    return "style_transfer/output.png"


def neural_style_transfer(model, content_image, style_image):
    print("-----------------loading model weights")
    image_type = ('jpg', 'jpeg', 'png')
    output = None
    if content_image[-3:] in image_type:
        # if "scream" in style_image:
        #     model.load_weights("style_transfer/models/scream/")
        # else:
        #     model = tf.keras.models.load_model(
        #         "style_transfer/models/arbitrary/")
        model = tf.keras.models.load_model(
            "style_transfer/models/arbitrary/")

        print("-----------------transferring style")
        content = load_image(content_image, image_size=(1024, 1024))
        style = load_image(style_image, (256, 256))

        # if "scream" in style_image:
        #     image = model(content)
        #     image = clip_0_1(image)
        #     output = tensor_to_image(image)
        # else:
        #     output = model(tf.constant(content),
        #                    tf.constant(style))[0][0]
        output = model(tf.constant(content), tf.constant(style))[0][0]
        print("-----------------saving generated image")
        output_path = save_result(output)
        print(f"-----------------generated image saved at {output_path}")
        return output_path
