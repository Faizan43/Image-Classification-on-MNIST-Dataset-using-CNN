from keras.models import Sequential
from keras.layers import Dense,Flatten,MaxPooling2D,Conv2D
from keras.utils.np_utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

(X_train,y_train),(X_test,y_test)=mnist.load_data()
width,height=28,28
X_train=X_train.reshape(60000,width*height)
X_test=X_test.reshape(10000,width*height)


X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train/=255.0
X_test/=255.0

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

model=Sequential()

model.add(Conv2D(32,kernel_size=(5,5),input_shape=(28,28,1),padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1024,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test))
history=model.evaluate(X_test,y_test)

