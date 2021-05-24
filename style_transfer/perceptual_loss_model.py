import tensorflow as tf
from .utils import vgg_layers, gram_matrix


class VGG19LossModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(VGG19LossModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(feature_map)
                         for feature_map in style_outputs]

        style_dict = {
            style_name: value for style_name, value in zip(self.style_layers, style_outputs)
        }

        content_dict = {
            content_name: value for content_name, value in zip(self.content_layers, content_outputs)
        }

        return {
            'content': content_dict,
            'style': style_dict
        }
