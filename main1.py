import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tkinter import filedialog


image_dir="augmented-images/"
img =  filedialog.askopenfilename()
head_tail = os.path.split(img)
fileNo=head_tail[1].split('.')
img = cv2.imread(head_tail[0]+'/'+fileNo[0]+'.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,150,255,cv2.THRESH_OTSU)

ret, markers = cv2.connectedComponents(thresh)

#Get the area taken by each component. Ignore label 0 since this is the background.
marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
#Get label of largest component by area
largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
#Get pixels which correspond to the brain
brain_mask = markers==largest_component

brain_out = img.copy()
#In a copy of the original image, clear those pixels that don't correspond to the brain
brain_out[brain_mask==False] = (0,0,0)

gray = cv2.cvtColor(brain_out, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
# extreme points
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# crop new image out of the original image using the four extreme points (left, right, top, bottom)
new_image = thresh[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
plt.imshow(img)
plt.title('Original Image')
plt.show()
plt.imshow(brain_out)
plt.title('Skull Removal')
plt.show()
plt.imshow(thresh, cmap='gray')
plt.title('Segmented Image')
plt.show()
plt.imshow(new_image, cmap='gray')
plt.title('Cropped Image')
plt.show()


encoder = OneHotEncoder()
encoder.fit([[0], [1]])

data = []
paths = []
result = []

for r, d, f in os.walk(r"C:\Users\vishwa\Downloads\BRAIN TUMOR DETECTION\brain_tumor_dataset\yes"):

    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))
             
for path in paths:
    img = Image.open(path)
    img = img.resize((65,65))
    img = np.array(img)
    if(img.shape == (65,65,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())
# This cell updates result list for images without tumor

paths = []
for r, d, f in os.walk(r"C:\Users\vishwa\Downloads\BRAIN TUMOR DETECTION\brain_tumor_dataset\no"):

    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))
            print(paths)
for path in paths:
    img = Image.open(path)
    img = img.resize((65,65))
    img = np.array(img)
    if(img.shape == (65,65,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())
data = np.array(data)
print(data.shape)

result = np.array(result)
result = result.reshape(140,2)

x_train,x_test,y_train,y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=0)

#cnn

model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(65, 65, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer='Adamax', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train, epochs = 30, batch_size = 40, verbose = 1)

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Accuracy'], loc='upper right')
plt.show()
history=model.history.history
cnn=model.evaluate(x_test,y_test,verbose=1)[1]*100
print('CNN accuracy is:',cnn,'%')

model.save('model.h5')

from keras.models import load_model
model = load_model('model.h5')

def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'
from matplotlib.pyplot import imshow
from tkinter import filedialog
import PIL

Image = filedialog.askopenfilename()
img = PIL.Image.open(Image)
x = np.array(img.resize((65,65)))
x = x.reshape(1,65,65,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Confidence This Is ' + names(classification))

