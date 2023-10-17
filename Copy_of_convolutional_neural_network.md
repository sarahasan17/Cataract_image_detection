# Convolutional Neural Network

### Importing the libraries


```python
#DeepLearning_convolutional_neural_network.ipynb
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
```


```python
tf.__version__
```




    '2.14.0'



## Part 1 - Data Preprocessing

### Preprocessing the Training set


```python
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)#last 3 are usually feature scaling
training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
```

    Found 491 images belonging to 2 classes.
    

### Preprocessing the Test set


```python
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

```

    Found 121 images belonging to 2 classes.
    

## Part 2 - Building the CNN

### Initialising the CNN


```python
cnn=tf.keras.models.Sequential()
```

### Step 1 - Convolution


```python
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
```

### Step 2 - Pooling


```python
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
```

### Adding a second convolutional layer


```python
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
```

### Step 3 - Flattening


```python
cnn.add(tf.keras.layers.Flatten())
```

### Step 4 - Full Connection


```python
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))#relu-> rectifier activation function till we use a
```

### Step 5 - Output Layer


```python
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
```

## Part 3 - Training the CNN

### Compiling the CNN


```python
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

### Training the CNN on the Training set and evaluating it on the Test set


```python
cnn.fit(x=training_set,validation_data=test_set,epochs=25)
```

    Epoch 1/25
    16/16 [==============================] - 27s 2s/step - loss: 0.7185 - accuracy: 0.5642 - val_loss: 0.6443 - val_accuracy: 0.5868
    Epoch 2/25
    16/16 [==============================] - 24s 2s/step - loss: 0.5923 - accuracy: 0.7434 - val_loss: 0.4098 - val_accuracy: 0.9008
    Epoch 3/25
    16/16 [==============================] - 25s 2s/step - loss: 0.4565 - accuracy: 0.8106 - val_loss: 0.3101 - val_accuracy: 0.8264
    Epoch 4/25
    16/16 [==============================] - 28s 2s/step - loss: 0.4282 - accuracy: 0.7963 - val_loss: 0.3311 - val_accuracy: 0.8512
    Epoch 5/25
    16/16 [==============================] - 24s 2s/step - loss: 0.3840 - accuracy: 0.8330 - val_loss: 0.2825 - val_accuracy: 0.9008
    Epoch 6/25
    16/16 [==============================] - 24s 1s/step - loss: 0.3349 - accuracy: 0.8350 - val_loss: 0.2753 - val_accuracy: 0.8926
    Epoch 7/25
    16/16 [==============================] - 23s 1s/step - loss: 0.3397 - accuracy: 0.8513 - val_loss: 0.2800 - val_accuracy: 0.8926
    Epoch 8/25
    16/16 [==============================] - 25s 2s/step - loss: 0.3178 - accuracy: 0.8595 - val_loss: 0.2398 - val_accuracy: 0.9091
    Epoch 9/25
    16/16 [==============================] - 23s 1s/step - loss: 0.2933 - accuracy: 0.8697 - val_loss: 0.2419 - val_accuracy: 0.9008
    Epoch 10/25
    16/16 [==============================] - 23s 1s/step - loss: 0.3128 - accuracy: 0.8615 - val_loss: 0.2294 - val_accuracy: 0.9339
    Epoch 11/25
    16/16 [==============================] - 23s 1s/step - loss: 0.2545 - accuracy: 0.8900 - val_loss: 0.2020 - val_accuracy: 0.9174
    Epoch 12/25
    16/16 [==============================] - 23s 1s/step - loss: 0.2485 - accuracy: 0.8839 - val_loss: 0.2219 - val_accuracy: 0.8926
    Epoch 13/25
    16/16 [==============================] - 23s 1s/step - loss: 0.2318 - accuracy: 0.9022 - val_loss: 0.1994 - val_accuracy: 0.9256
    Epoch 14/25
    16/16 [==============================] - 23s 1s/step - loss: 0.2368 - accuracy: 0.8859 - val_loss: 0.2595 - val_accuracy: 0.9008
    Epoch 15/25
    16/16 [==============================] - 24s 2s/step - loss: 0.2913 - accuracy: 0.8778 - val_loss: 0.3805 - val_accuracy: 0.8347
    Epoch 16/25
    16/16 [==============================] - 24s 1s/step - loss: 0.2839 - accuracy: 0.8717 - val_loss: 0.2425 - val_accuracy: 0.9339
    Epoch 17/25
    16/16 [==============================] - 24s 2s/step - loss: 0.2447 - accuracy: 0.8961 - val_loss: 0.2900 - val_accuracy: 0.9339
    Epoch 18/25
    16/16 [==============================] - 24s 2s/step - loss: 0.2341 - accuracy: 0.9002 - val_loss: 0.1947 - val_accuracy: 0.9421
    Epoch 19/25
    16/16 [==============================] - 24s 2s/step - loss: 0.2141 - accuracy: 0.9084 - val_loss: 0.1928 - val_accuracy: 0.9256
    Epoch 20/25
    16/16 [==============================] - 23s 1s/step - loss: 0.1953 - accuracy: 0.9206 - val_loss: 0.2149 - val_accuracy: 0.9421
    Epoch 21/25
    16/16 [==============================] - 23s 1s/step - loss: 0.1692 - accuracy: 0.9470 - val_loss: 0.1853 - val_accuracy: 0.9339
    Epoch 22/25
    16/16 [==============================] - 23s 1s/step - loss: 0.1610 - accuracy: 0.9369 - val_loss: 0.1934 - val_accuracy: 0.9256
    Epoch 23/25
    16/16 [==============================] - 23s 2s/step - loss: 0.1775 - accuracy: 0.9308 - val_loss: 0.2096 - val_accuracy: 0.9256
    Epoch 24/25
    16/16 [==============================] - 23s 1s/step - loss: 0.1769 - accuracy: 0.9226 - val_loss: 0.2899 - val_accuracy: 0.9256
    Epoch 25/25
    16/16 [==============================] - 23s 1s/step - loss: 0.1514 - accuracy: 0.9369 - val_loss: 0.2299 - val_accuracy: 0.9256
    




    <keras.src.callbacks.History at 0x1f5fa041090>



## Part 4 - Making a single prediction


```python
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/cataract_or_normal_eye_1.png',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'Normal Eye'
else:
  prediction = 'Cataract'
```

    1/1 [==============================] - 0s 33ms/step
    


```python
print(prediction)
```

    Cataract
    


```python
test_image = image.load_img('dataset/image_290.png',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'Normal Eye'
else:
  prediction = 'Cataract'
```

    1/1 [==============================] - 0s 31ms/step
    


```python
print(prediction)
```

    Normal Eye
    


```python

```
