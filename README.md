# Background
This repo is used to store sample code from this article as [Keras - MNIST 手寫數字辨識使用 CNN](http://puremonkey2010.blogspot.com/2017/07/toolkit-keras-mnist-cnn.html)

The sample codes written in his repo can run in Python3 and you can install the dependent packages this way:
```console
# virtualenv -p python3 env
# source env/bin/activate
# pip install -r requirements.txt
```

Then you can run the training by execute `train.py`:
```
# ./train.py
...
10000/10000 [==============================] - 1s 130us/step

        [Info] Accuracy of testing data = 99.0%
```
It will load in the serialized model `keras_mnist_cnn.h5` if it exists. If you want to retrain the model. Please modify `train.py` and setup below setting to `False` (default is `True`):
```python
USE_SM_IF_EXIST = False
```

If you want to predict your image(s), please save them into jpg and store into folder `datas/mnist` with size as `28x28`. Then you can use another utility `test_my_images.py` to do the prediction:
```console
# ./test_my_images.py
...
datas/mnist/my_0.jpg is predicted as 0
datas/mnist/my_1.jpg is predicted as 1
datas/mnist/my_2.jpg is predicted as 2
datas/mnist/my_3.jpg is predicted as 3
datas/mnist/my_4.jpg is predicted as 7
datas/mnist/my_5.jpg is predicted as 5
datas/mnist/my_6.jpg is predicted as 5
datas/mnist/my_7.jpg is predicted as 7
datas/mnist/my_8.jpg is predicted as 3
datas/mnist/my_9.jpg is predicted as 7
``` 
