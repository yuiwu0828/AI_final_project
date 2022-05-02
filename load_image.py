import cv2, os, sys
import numpy as np

os.chdir(sys.path[0])

def load_image(dataPath):
    dataset = []
    folder = os.listdir(dataPath)
    ID = 0
    for folder_name in folder:
        files = os.listdir(dataPath + '/' + folder_name)
        count = 0
        for name in files:
            img = cv2.imread(dataPath + '/' + folder_name + '/' + name)
            img = cv2.resize(img, (36, 16))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            element = (np.array(img), 1)
            dataset.append((element, ID))
            count = count + 1
            if count >= 400:
                break
        ID = ID + 1

    return dataset
"""
if __name__ == '__main__':
    dataPath = 'C:/Users/User/Desktop/final_project/used/Train'
    load_image(dataPath)
    print('done')
"""
