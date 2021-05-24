import tensorflow as tf
from .layers import InstanceNormConv2D, Residual, ResizableConv2D


class TransformerNet(tf.keras.models.Model):
    def __init__(self):
        super(TransformerNet, self).__init__()
        print("-----------------building model architecture")
        self.conv1 = InstanceNormConv2D(filters=32, kernel=9, stride=1)
        self.conv2 = InstanceNormConv2D(filters=64, kernel=3, stride=2)
        self.conv3 = InstanceNormConv2D(filters=128, kernel=3, stride=2)

        self.res1 = Residual(filters=128, kernel=3, stride=1)
        self.res2 = Residual(filters=128, kernel=3, stride=1)
        self.res3 = Residual(filters=128, kernel=3, stride=1)
        self.res4 = Residual(filters=128, kernel=3, stride=1)
        self.res5 = Residual(filters=128, kernel=3, stride=1)

        self.resize_conv1 = ResizableConv2D(filters=64, kernel=3, stride=2)
        self.resize_conv2 = ResizableConv2D(filters=32, kernel=3, stride=2)
        self.conv4 = InstanceNormConv2D(filters=3, kernel=9, stride=1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.resize_conv1(x)
        x = self.resize_conv2(x)
        x = self.conv4(x, relu=False)

        return (tf.keras.activations.tanh(x) * 150 + 255.0 / 2)
