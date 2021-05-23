import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras import Sequential
from tensorflow.python.keras.engine.training import Model

class MusicGenerationModel:
    def __init__(self, artist, n_vocab, input_shape):
        self.model = None
        self.artist = artist
        self.vocab = n_vocab
        self.input_shape = input_shape

    def init_model_architecture(self):
        print("-----------------compiling model architecture")
        model = Sequential()
        model.add(LSTM(512, input_shape=self.input_shape, return_sequences=True, recurrent_dropout=0))
        model.add(LSTM(512, return_sequences=True, recurrent_dropout=0))
        model.add(LSTM(512))
        if self.artist == "schubert":
            model.add(BatchNormalization())
            model.add(Dropout(0.3))

        model.add(Dense(256, activation="relu"))

        if self.artist == "schubert":
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
        
        if self.artist != "schubert":
            model.add(Dense(128, activation="relu"))
        model.add(Dense(self.vocab, activation="softmax"))
        self.model = model

    def load_model_weights(self):
        print("-----------------loading model weights")
        if self.model is not None:
            self.model.load_weights(f"music_generation/models/{self.artist}.h5")
        else:
            print("Could not load model weights")
            self.model = None
        return self.model

    def print_model_summary(self):
        if self.model is not None:
            print(self.model.summary())
        else:
            print("Model does not exist")