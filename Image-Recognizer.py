#First Part
import tensorflow as tf
fashion_mnist=tf.keras.datasets.fashion_mnist
(x_train,y_train) , (x_test,y_test)=fashion_mnist.load_data() #Loading MNIST Datasets
x_train=tf.keras.utils.normalize(x_train,axis=1)# Normalizing training data
x_test=tf.keras.utils.normalize(x_test,axis=1)# Normalizing test data
models=tf.keras.models.Sequential() #sequential model
models.add(tf.keras.layers.Flatten()) #InputLayer
models.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #1st HiddenLayer
models.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #2nd HiddenLayer
models.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) #OutputLayer
models.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
models.fit(x_train,y_train,epochs=5)

#Second Part
loss1,acc=models.evaluate(x_test,y_test)
print(loss1,acc)

#Third Part
predict1=models.predict([x_test])
print(predict1)

#Fourth Part
list1=['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
import numpy as nm
a=nm.argmax(predict1[18])
print(list1[a])

#Fifth Part
import matplotlib.pyplot as plt
plt.imshow(x_test[18],cmap=plt.cm.binary)
plt.show()

