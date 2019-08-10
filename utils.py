import os  
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def isDisplayAvl():  
    return 'DISPLAY' in os.environ.keys()  
  

def jpg2mnist(image_fp):
    r'''
    Read in image file (jpg) and transform into MNIST format
    '''
    img_data = matplotlib.pyplot.imread(image_fp)
    img_mnist_fmt = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            img_mnist_fmt[i, j] = 255 - int(np.mean(img_data[i, j][:3]))

    return img_mnist_fmt

def plot_image(image):  
    fig = plt.gcf()  
    fig.set_size_inches(2,2)  
    plt.imshow(image, cmap='binary')  
    plt.show()  
  
def plot_images_labels_predict(images, labels, prediction, idx, num=10):  
    fig = plt.gcf()  
    fig.set_size_inches(12, 14)  
    if num > 25: num = 25  
    for i in range(0, num):  
        ax=plt.subplot(5,5, 1+i)  
        ax.imshow(images[idx], cmap='binary')  
        title = "l=" + str(labels[idx])  
        if len(prediction) > 0:  
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))  
        else:  
            title = "l={}".format(str(labels[idx]))  
        ax.set_title(title, fontsize=10)  
        ax.set_xticks([]); ax.set_yticks([])  
        idx+=1  
    plt.show()  
  
def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  
