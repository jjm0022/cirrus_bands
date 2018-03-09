

from sklearn import svm, metrics
from PIL import Image
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import random
import numpy as np

def load_img(path, target_size=(256,256)): 
    '''Load an image into PIL format.
    # Arguments
      path: path to image file
      target_size: None (default to original size)
          or (img_height, img_width)
    '''
    img = Image.open(path)
    img = img.convert('L')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


def img_to_array(img):
    '''Converts a PIL image object into a numpy array
    # Arguments
      x: a numpy array/martix
    '''
    x = np.asarray(img, dtype=np.uint8)
    # To apply a classifier on this data, we need to flatten the image, to
	# turn the data in a (samples, feature) matrix:
    x = x.flatten()
    return x


def array_to_image(x):
    '''returns a PIL Image object
    # Arguments
      x: a numpy array/martix
    '''
    image = Image.fromarray(x)
    return image


def load_data(path):
    images = []
    labels = []
    for path, dirs, files, in os.walk(path):
        if files:
            class_ = path.split('/')[-1]
            if class_ == 'bands':
                label = 1
            else:
                label = -1
            for f in files:
                img = load_img(os.path.join(path, f))
                x = img_to_array(img)
                images.append(x)
                labels.append(label)

    images_and_labels = (images, labels)
    return images_and_labels


def list_shuffle(l, x=5):
    for i in range(x):
        random.shuffle(l)
    return l


# path to the train images
train_path = '/home/jmiller/Dropbox/school/cnn_stuff/soni_images/paper_data/train'
test_path = '/home/jmiller/Dropbox/school/cnn_stuff/soni_images/paper_data/test'
print('Loading Training Images...\n')
train_images, train_labels = load_data(train_path)
#shuffle the images and labels for the train set
n_samples = len(train_images)
l = range(n_samples)
l = list_shuffle(l)
X = [train_images[i] for i in l]
Y = [train_labels[i] for i in l]
print('Loading Test Images...\n')
test_images, test_labels = load_data(test_path)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

features = ['auto', 500, 1000]
depth = [5, 10, 20]
estimators = [100, 500, 1000]
parameters = dict(max_features=features, max_depth=depth, n_estimators=estimators)


#classifier = RandomForestClassifier(verbose=1, n_jobs=3)
#clf = GridSearchCV(classifier, parameters, scoring='f1')
print('Training classifier...\n')
# We learn the digits on the first half of the digits
clf.fit(X, Y)
print('Evaluating...\n')
# Now predict the value of the digit on the second half:
expected = test_labels
predicted = classifier.predict(test_images)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
