from tensorflow import pad, constant
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Lambda, ZeroPadding2D
from keras.models import Model
from keras.initializers import Constant
from tensorflow import pad, constant


class SoccerMap: 

    def __init__(self, field_dimen):
        self.field_dimen = field_dimen

    def symmetric_pad(self, x):
        paddings = constant([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        return pad(x, paddings, "SYMMETRIC")
    
    def get_model(self):
        
        pass_input1x = Input(shape=(self.field_dimen[0], self.field_dimen[1], 13), name='pass_input')

        x = Conv2D(32, (5, 5), activation='relu')(pass_input1x)
        # x = Lambda(self.symmetric_pad)(x)
        x = ZeroPadding2D((4,4))(x)
        x = Conv2D(64, (5, 5), activation='relu')(x) # 108x72x1

        #Prediction
        x = Conv2D(32, (1,1), activation='relu')(x)
        x = Conv2D(1, (1,1), activation='linear')(x) #108x72x1

        y = MaxPooling2D((2, 2), strides=(2, 2))(x) # 54x36x1
        y = Conv2D(32, (5, 5), activation='relu')(y) 
        # y = Lambda(self.symmetric_pad)(y)
        y = ZeroPadding2D((4,4))(y)
        y = Conv2D(64, (5, 5), activation='relu')(y) # 54x36x1

        #Prediction
        y = Conv2D(32, (1,1), activation='relu')(y)
        y = Conv2D(1, (1,1), activation='linear')(y) # 54x36x1

        z = MaxPooling2D((2, 2), strides=(2, 2))(y) # 27x18x1
        z = Conv2D(32, (5, 5), activation='relu')(z)
        # z = Lambda(self.symmetric_pad)(z)
        z = ZeroPadding2D((4,4))(z)
        z = Conv2D(64, (5, 5), activation='relu')(z) # 27x18x1

        #Prediction
        z = Conv2D(32, (1,1), activation='relu')(z) 
        z = Conv2D(1, (1,1), activation='linear')(z) # 27x18x1

        #Upsampling
        z = UpSampling2D(size=(2, 2), interpolation="nearest")(z) # 54x36x1
        z = Conv2D(32, (3, 3), activation='relu')(z) # 52x34x1
        z = ZeroPadding2D((2,2))(z)
        z = Conv2D(1, (3, 3), activation='linear')(z) #50x32

        #Concatenation
        yz = Concatenate()([y,z])

        #Fusion
        yz = Conv2D(1, (1, 1))(yz)

        #Upsampling
        yz = UpSampling2D(size=(2, 2), interpolation="nearest")(yz)
        yz = Conv2D(32, (3, 3), activation='relu')(yz)
        yz = ZeroPadding2D((2,2))(yz)
        yz = Conv2D(1, (3, 3), activation='linear')(yz)

        #Concatenation
        xyz = Concatenate()([x,yz])

        #Fusion 
        xyz = Conv2D(1, (1, 1))(xyz)

        #Prediction
        xyz = Conv2D(32, (1,1), activation='relu')(xyz)
        xyz = Conv2D(1, (1,1), activation='linear')(xyz)

        #sigmoid activation
        # out = Conv2D(1, (1,1), activation='sigmoid', kernel_initializer=Constant(avg_completion_rate))(x)
        out = Conv2D(1, (1,1), activation='sigmoid')(xyz)
        
        model = Model([pass_input1x], out)

        self.model = model

    def compile(self, loss, optimizer):
        self.get_model()
        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, X, y, X_val, y_val, eps): 
        self.fitted_model = self.model.fit(X, y,
             epochs=eps, 
             validation_data=(X_val, y_val)
        )
