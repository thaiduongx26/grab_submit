from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Lambda, Dropout

img_width, img_height = 224, 224

def createModel(num_classes):
    input_tensor = Input(shape=(img_height, img_width, 3))
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    net = base_model.output
    net = GlobalAveragePooling2D()(net)
    # net = Dense(1024, activation='relu')(net)
    net = Dropout(0.2)(net)
    net = Dense(num_classes, activation='softmax')(net)

    model = Model(base_model.input, net)
    for layer in base_model.layers:
        layer.trainable = True
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model