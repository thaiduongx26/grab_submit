from model import createModel
import preprocess
import cv2
from PIL import Image
import numpy as np
import sys
import argparse
import os
import pandas as pd

weights_path = "models/model.80-0.86.hdf5"
num_classes = 196
classes = {0: '1', 1: '10', 2: '100', 3: '101', 4: '102', 5: '103', 6: '104', 7: '105', 8: '106', 9: '107', 10: '108', 11: '109', 12: '11', 13: '110', 14: '111', 15: '112', 16: '113', 17: '114', 18: '115', 19: '116', 20: '117', 21: '118', 22: '119', 23: '12', 24: '120', 25: '121', 26: '122', 27: '123', 28: '124', 29: '125', 30: '126', 31: '127', 32: '128', 33: '129', 34: '13', 35: '130', 36: '131', 37: '132', 38: '133', 39: '134', 40: '135', 41: '136', 42: '137', 43: '138', 44: '139', 45: '14', 46: '140', 47: '141', 48: '142', 49: '143', 50: '144', 51: '145', 52: '146', 53: '147', 54: '148', 55: '149', 56: '15', 57: '150', 58: '151', 59: '152', 60: '153', 61: '154', 62: '155', 63: '156', 64: '157', 65: '158', 66: '159', 67: '16', 68: '160', 69: '161', 70: '162', 71: '163', 72: '164', 73: '165', 74: '166', 75: '167', 76: '168', 77: '169', 78: '17', 79: '170', 80: '171', 81: '172', 82: '173', 83: '174', 84: '175', 85: '176', 86: '177', 87: '178', 88: '179', 89: '18', 90: '180', 91: '181', 92: '182', 93: '183', 94: '184', 95: '185', 96: '186', 97: '187', 98: '188', 99: '189', 100: '19', 101: '190', 102: '191', 103: '192', 104: '193', 105: '194', 106: '195', 107: '196', 108: '2', 109: '20', 110: '21', 111: '22', 112: '23', 113: '24', 114: '25', 115: '26', 116: '27', 117: '28', 118: '29', 119: '3', 120: '30', 121: '31', 122: '32', 123: '33', 124: '34', 125: '35', 126: '36', 127: '37', 128: '38', 129: '39', 130: '4', 131: '40', 132: '41', 133: '42', 134: '43', 135: '44', 136: '45', 137: '46', 138: '47', 139: '48', 140: '49', 141: '5', 142: '50', 143: '51', 144: '52', 145: '53', 146: '54', 147: '55', 148: '56', 149: '57', 150: '58', 151: '59', 152: '6', 153: '60', 154: '61', 155: '62', 156: '63', 157: '64', 158: '65', 159: '66', 160: '67', 161: '68', 162: '69', 163: '7', 164: '70', 165: '71', 166: '72', 167: '73', 168: '74', 169: '75', 170: '76', 171: '77', 172: '78', 173: '79', 174: '8', 175: '80', 176: '81', 177: '82', 178: '83', 179: '84', 180: '85', 181: '86', 182: '87', 183: '88', 184: '89', 185: '9', 186: '90', 187: '91', 188: '92', 189: '93', 190: '94', 191: '95', 192: '96', 193: '97', 194: '98', 195: '99'}

def load_model(num_classes):
    model = createModel(num_classes)
    model.load_weights(weights_path, by_name=True)
    return model

def predict(image, model):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = preprocess.preprocess_image(image)
    img = img / 255.
    preds = model.predict(np.reshape(img, (1, 224, 224, 3)))
    prob = np.max(preds)
    class_id = np.argmax(preds)
    # print(classes[class_id])
    return classes[class_id]

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
            print("Label of {}".format(argv.image) + " is {}".format(predict(image, model)))
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
            data.to_csv(index=False)
            print(data)

