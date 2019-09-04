import numpy as np
np.random.seed(3)
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(
    'C:/Users/CPB06GameN/Desktop/동아리/detection_dataset/train',
    target_size=(230,230),
    batch_size=6,
    class_mode='categorical'
)
test_datagen=ImageDataGenerator(rescale=1./255)
test_generator=test_datagen.flow_from_directory(
    'C:/Users/CPB06GameN/Desktop/동아리/detection_dataset/test',
    target_size=(230,230),
    batch_size=6,
    class_mode='categorical'
)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
model=Sequential()
model.add(Conv2D(16,kernel_size=(3,3),
                activation='relu',
                input_shape=(230,230,3)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(6,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=15,
    epochs=200,
    validation_data=test_generator,
    validation_steps=6
)
scores = model.evaluate_generator(
            test_generator, 
            steps = 5)

print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
output = model.predict_generator(
            test_generator, 
            steps = 5)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})