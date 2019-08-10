#!/usr/bin/env python
import numpy as np
from keras.models import load_model
from train import SERIALIZED_MODEL_NAME 

class MyCfr:
    def __init__(self):
        self.model = load_model(SERIALIZED_MODEL_NAME)

    def p(self, img_data):
        r'''
        Making prediction on given image data as numpy array

        :img_data: numpy array (28x28)
        '''
        input_fmt = img_data.reshape(1, 28, 28, 1).astype('float32') / 255
        return self.model.predict_classes(input_fmt)[0]
