from keras.initializers.initializers_v2 import Constant
from tensorflow import pad, constant
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Lambda, ZeroPadding2D, Dense
from keras.models import Model
from tensorflow import pad, constant
import keras.backend as K


class SoccerMap:

    def __init__(self, field_dimen):
        self.field_dimen = field_dimen
        self.model, self.full = self.get_model()

    def symmetric_pad(self, x):
        paddings = constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return pad(x, paddings, "SYMMETRIC")

    def get_model(self):
        pass_input = Input(shape=(self.field_dimen[0], self.field_dimen[1], 15), name='pass_input')
        dest_input = Input(shape=(self.field_dimen[0], self.field_dimen[1], 1), name='dest_input')

        x = Conv2D(32, (5, 5), activation='relu')(pass_input)
        # x = Lambda(self.symmetric_pad)(x)
        x = ZeroPadding2D((4, 4))(x)
        x = Conv2D(64, (5, 5), activation='relu')(x)  # 108x72x1

        # Prediction
        x = Conv2D(32, (1, 1), activation='relu')(x)
        x = Conv2D(1, (1, 1), activation='linear')(x)  # 108x72x1

        y = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 54x36x1
        y = Conv2D(32, (5, 5), activation='relu')(y)
        # y = Lambda(self.symmetric_pad)(y)
        y = ZeroPadding2D((4, 4))(y)
        y = Conv2D(64, (5, 5), activation='relu')(y)  # 54x36x1

        # Prediction
        y = Conv2D(32, (1, 1), activation='relu')(y)
        y = Conv2D(1, (1, 1), activation='linear')(y)  # 54x36x1

        z = MaxPooling2D((2, 2), strides=(2, 2))(y)  # 27x18x1
        z = Conv2D(32, (5, 5), activation='relu')(z)
        # z = Lambda(self.symmetric_pad)(z)
        z = ZeroPadding2D((4, 4))(z)
        z = Conv2D(64, (5, 5), activation='relu')(z)  # 27x18x1

        # Prediction
        z = Conv2D(32, (1, 1), activation='relu')(z)
        z = Conv2D(1, (1, 1), activation='linear')(z)  # 27x18x1

        # Upsampling
        z = UpSampling2D(size=(2, 2), interpolation="nearest")(z)  # 54x36x1
        z = Conv2D(32, (3, 3), activation='relu')(z)  # 52x34x1
        z = ZeroPadding2D((2, 2))(z)
        z = Conv2D(1, (3, 3), activation='linear')(z)  # 50x32

        # Concatenation
        yz = Concatenate()([y, z])

        # Fusion
        yz = Conv2D(1, (1, 1))(yz)

        # Upsampling
        yz = UpSampling2D(size=(2, 2), interpolation="nearest")(yz)
        yz = Conv2D(32, (3, 3), activation='relu')(yz)
        yz = ZeroPadding2D((2, 2))(yz)
        yz = Conv2D(1, (3, 3), activation='linear')(yz)

        # Concatenation
        xyz = Concatenate()([x, yz])

        # Fusion
        xyz = Conv2D(1, (1, 1))(xyz)

        # Prediction
        xyz = Conv2D(32, (1, 1), activation='relu')(xyz)
        xyz = Conv2D(1, (1, 1), activation='linear')(xyz)

        # sigmoid activation
        # out = Conv2D(1, (1,1), activation='sigmoid', kernel_initializer=Constant(avg_completion_rate))(x)
        out_field = Dense(1, activation='sigmoid')(xyz)

        combined = Concatenate()([out_field, dest_input])

        out_pixel = Lambda(self.pixel_layer)(combined)

        full = Model([pass_input, dest_input], out_pixel)
        model = Model([pass_input, dest_input], combined)
        return model, full

    def compile(self, loss, optimizer):
        self.model.compile(loss=loss, optimizer=optimizer)
        self.full.compile(loss=loss, optimizer=optimizer)

    def pixel_layer(self, x):
        surface = x[:, :, :, 0]
        mask = x[:, :, :, 1]
        masked = surface * mask
        value = K.sum(masked, axis=(2,1))
        return value
