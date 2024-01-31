#data preprocessing 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
fashion = keras.datasets.fashion_mnist(xtrain, ytrain), (xtest, ytest) = fashion.load_data

#a look at one of the images
imgIndex = 9
image = xtrain[imgIndex]
print("Image Label:", ytrain[imgIndex])
plt.imshow(image)

#a look at both training and test data
print(xtrain.shape)
print(xtest.shape)

#build a neural network architecture 
model = keras.models.Sequential([
       keras.layers.Flatten(input_shape= [28, 28])
       keras.layers.Dense(300, activation="relu"),
       keras.layers.Dense(100, activation="relu"),
       keras.layers.Dense(10, activation="softmax)])

print(model.summary())                  

#split the data into training  and test validation data                      
xvalid, xtrain = xtrain[:5000]/255.0, xtrain[5000:]/255.0
yvalid, ytrain = ytrain[:5000]/255.0, ytrain[5000:]

#train the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ([accuracy])
history = model.fit(xtrain, ytrain, epoch=30,
                    validation_data = (xvalid, yvalid))
              
              
#predictions or testing trained model                        
new = xtest[:5]
predictions = model.predict(new)
print(predictions)

#look at predicted classes
classes = np.argmax(predictions, axis=1)
print(classes)


      
