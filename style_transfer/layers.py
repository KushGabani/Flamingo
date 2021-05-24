import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.beta = tf.Variable(tf.zeros([input_shape[3]]))
        self.gamma = tf.Variable(tf.ones([input_shape[3]]))

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean),
                      tf.sqrt(tf.add(var, self.epsilon)))

        return self.gamma * x + self.beta


class InstanceNormConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(InstanceNormConv2D, self).__init__()
        pad = kernel // 2
        self.padding = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        self.conv = tf.keras.layers.Conv2D(
            filters, kernel, stride, use_bias=False, padding='valid')
        self.instance_norm = InstanceNormalization()

    def call(self, inputs, relu=True):
        x = tf.pad(inputs, self.padding, mode='REFLECT')
        x = self.conv(x)
        x = self.instance_norm(x)

        if relu:
            x = tf.keras.layers.Activation("relu")(x)
        return x


class Residual(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(Residual, self).__init__()
        self.conv1 = InstanceNormConv2D(filters, kernel, stride)
        self.conv2 = InstanceNormConv2D(filters, kernel, stride)

    def call(self, inputs):
        x = self.conv1(inputs)
        return inputs + self.conv2(x, relu=False)


class ResizableConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(ResizableConv2D, self).__init__()
        self.conv = InstanceNormConv2D(filters, kernel, stride)
        self.stride = stride

    def call(self, inputs):
        height = inputs.shape[1] * self.stride * 2
        width = inputs.shape[2] * self.stride * 2
        x = tf.image.resize(
            inputs, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = self.conv(x)
        return x


class InstanceNormConv2DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(InstanceNormConv2DTranspose, self).__init__()
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            filters, kernel, stride, padding="same")
        self.instance_norm = InstanceNormalization()

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        x = self.instance_norm(x)
        return tf.keras.layers.Activation("relu")
