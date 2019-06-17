from model import createModel
import preprocess
import cv2
from PIL import Image
import numpy as np
import sys
import argparse
import os
import pandas as pd

weights_path = "models/model.24-0.84.hdf5"
num_classes = 196

def load_model(num_classes):
    model = createModel(num_classes)
    model.load_weights(weights_path, by_name=True)
    return model

def predict(image, model):
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = preprocess.preprocess_image(image)
    img = img / 255.
    # cv2.imwrite("fuck.jpg", img)
    # print(img)
    preds = model.predict(np.reshape(img, (1, 224, 224, 3)))
    prob = np.max(preds)
    class_id = np.argmax(preds)
    print(class_id)
    print(prob)
    return class_id
if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    
    parser.add_argument(
        '--image', type=str, help='path to image test', default=""
    )

    parser.add_argument(
        '--testdir', type=str, help='path to test dir', default=""
    )

    argv = parser.parse_args()

    model = load_model(num_classes)

    if(argv.image != ""):
        if(os.path.isfile(argv.image)):
            image = cv2.imread(argv.image)
            print("Label of {}".format(argv.image) + "is {}".format(predict(image, model)))
        else:
            print("The image does not exist")

    if(argv.testdir != ""):
        filename = []
        result = []
        if(os.path.isdir(argv.testdir)):
            listimg = os.listdir(argv.testdir)
            for i in range(len(listimg)):
                image = cv2.imread((argv.testdir) + "/" + listimg[i])
                filename.append(listimg[i])
                res = predict(image, model)
                result.append(res)
            res_final = {"filename": filename, "result": result}
            data = pd.DataFrame(data=res_final)
            print(data)

