# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import time
import cv2

from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

import dataset as ds
import embeding as eb

# load modal
face_model = load_model('facenet_keras.h5')
print('Loaded Model')


# load faces
data = load('5-celebrity-faces-dataset.npz')
testX_faces = data['arr_2']
# load face embeddings
data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
print('Train Model')


camera_port = 0
camera = cv2.VideoCapture(camera_port)
time.sleep(0.1)  # If you don't wait, the image will be dark
return_value, image = camera.read()
# cv2.imwrite("opencv.png", image)
del(camera)  # so that others can use the camera as soon as possible

# plot
# pyplot.imshow(image)
# pyplot.show()

face = ds.extract_face_from_image(image)

# directory = '5-celebrity-faces-dataset/train/'
# subdir = 'ian_flynn'
# fileName = 'test.jpg'
# path = directory + subdir + '/' + fileName
# face = ds.extract_face(path)

# pyplot.imshow(face)
# pyplot.show()

embedding = eb.get_embedding(face_model, face)

# prediction for the face
samples = expand_dims(embedding, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

# plot for to see
pyplot.imshow(image)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()