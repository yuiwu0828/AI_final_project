import cv2, os, sys
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical

os.chdir(sys.path[0])

def load_image(dataPath):
    dataset = []
    labels = []
    folder = os.listdir(dataPath)
    ID = 0
    for folder_name in folder:
        files = os.listdir(dataPath + '/' + folder_name)
        count = 0
        for name in files:
            img = cv2.imread(dataPath + '/' + folder_name + '/' + name)
            img = cv2.resize(img, (36, 36))
            dataset.append(img_to_array(img))
            labels.append(ID)
            count = count + 1
        ID = ID + 1
    dataset = np.array(dataset, dtype='float32')/255.0
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=86)
    return dataset, labels


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    legendLoc = 'lower right' if(train=='acc') else 'upper right'
    plt.legend(['train', 'validation'], loc=legendLoc)
    plt.show()