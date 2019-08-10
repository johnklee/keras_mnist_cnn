#!/usr/bin/env python
import os
from utils import *
from predict import *

data_sources = 'datas/mnist'
mc = MyCfr()

for f in os.listdir(data_sources):
    if f.endswith('.jpg'):
        fp = os.path.join(data_sources, f)
        img_data = jpg2mnist(fp)
        print('{} is predicted as {}'.format(fp, mc.p(img_data)))
