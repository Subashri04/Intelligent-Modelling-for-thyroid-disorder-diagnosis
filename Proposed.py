
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121, VGG16, VGG19, MobileNetV3
from tensorflow.keras.utils import to_categorical
from numpy import argmax
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
import easygui

#==============================Input Data======================================
path = 'Dataset/'

# categories
categories = ['thyroid_cancer','thyroid_ditis','thyroid_hyper','thyroid_hypo','thyroid_nodule','thyroid_normal']

# Segmentation: Optimized Otsu's Method and Edge Detection
def optimized_otsu(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_image

# Display sample images from each category
for category in categories:
    fig, _ = plt.subplots(2, 2)
    fig.suptitle(category)
    fig.patch.set_facecolor('xkcd:white')
    for k, v in enumerate(os.listdir(path+category)[:4]):
        img = plt.imread(path+category+'/'+v)
        plt.subplot(2, 2, k+1)
        plt.axis('off')
        plt.imshow(img)

#==============================Image Preprocessing======================================
shape0 = []
shape1 = []

for category in categories:
    for files in os.listdir(path+category):
        shape0.append(plt.imread(path+category+'/'+ files).shape[0])
        shape1.append(plt.imread(path+category+'/'+ files).shape[1])

# initialize the data and labels
data = []
labels = []
imagePaths = []
HEIGHT = 65
WIDTH = 65
N_CHANNELS = 3

# grab the image paths and randomly shuffle them
for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k]) 

import random
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  
    data.append(image)
    
    label = imagePath[1]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Partition the data into training and testing splits using 80% for training and 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
trainY = to_categorical(trainY, 6)

#================================ Classification: Model Definition ====================================

def build_densenet():
    densenet = DenseNet121(weights='imagenet', include_top=False)
    input = Input(shape=(HEIGHT, WIDTH, N_CHANNELS))
    x = Conv2D(3, (3, 3), padding='same')(input)
    x = densenet(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(6, activation='softmax', name='root')(x)
    model = Model(input, output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model

def build_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, N_CHANNELS))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(6, activation='softmax', name='root')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model

def build_vgg19():
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, N_CHANNELS))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(6, activation='softmax', name='root')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model

def build_hybrid_densenet_mobilenet():
    base_model_1 = DenseNet121(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, N_CHANNELS))
    base_model_2 = MobileNetV3(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, N_CHANNELS))
    
    x1 = base_model_1.output
    x1 = GlobalAveragePooling2D()(x1)
    
    x2 = base_model_2.output
    x2 = GlobalAveragePooling2D()(x2)
    
    x = np.concatenate([x1, x2], axis=-1)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(6, activation='softmax', name='root')(x)
    
    model = Model(inputs=[base_model_1.input, base_model_2.input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model

#================================ Model Training ====================================
model = build_densenet()  # You can switch to `build_vgg16()` or `build_hybrid_densenet_mobilenet()`

history = model.fit(trainX, trainY, batch_size=32, epochs=5, verbose=1)

# Save the model
import tensorflow as tf
tf.keras.models.save_model(model, 'model.h5')

# Performance Graphs
train_loss = history.history['loss']
train_acc = history.history['accuracy']
plt.figure()
plt.plot(train_loss, label='Loss')
plt.plot(train_acc, label='Accuracy')
plt.title('Performance Plot')
plt.legend()
plt.show()

print("Accuracy of the CNN is:", model.evaluate(trainX, trainY)[1] * 100, "%")

#================================ Analytic Results ====================================
# Test set prediction and evaluation
pred = model.predict(testX)
predictions = argmax(pred, axis=1)

print('Classification Report')
cr = classification_report(testY, predictions, target_names=categories)
print(cr)

print('Confusion Matrix')
cm = confusion_matrix(testY, predictions)
print(cm)

# Confusion Matrix Plot
plt.figure()
plot_confusion_matrix(cm, figsize=(15, 15), class_names=categories, show_normed=True)
plt.title("Model confusion matrix")
plt.style.use("ggplot")
plt.show()

#=============================== Prediction on New Image ================================
# from tkinter import filedialog

# # Load a new image for prediction
# test_data = []
# Image = filedialog.askopenfilename()
# head_tail = os.path.split(Image)
# fileNo = head_tail[1].split('.')
# test_image_o = cv2.imread(Image)
# test_image = cv2.resize(test_image_o, (WIDTH, HEIGHT))

# # Normalize and prepare for prediction
# test_image_normalized = test_image / 255.0
# test_data = np.expand_dims(test_image_normalized, axis=0)

# # Predict
# pred = model.predict(test_data)
# predictions = argmax(pred, axis=1)
# print('Prediction: ' + categories[predictions[0]])

# # Display the image with prediction label
# fig = plt.figure()
# fig.patch.set_facecolor('xkcd:white')
# plt.title(categories[predictions[0]])
# plt.imshow(test_image_o)
# plt.show()
