#overfitting can be done regularizer and dropout

from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import fashion_mnist,cifar10
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt


(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#normalize 
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

#to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#architecture
model_base = Sequential()
model_base.add(Flatten(input_shape=(32,32,3)))
model_base.add(Dense(1024,activation='relu'))
model_base.add(Dense(512,activation='relu'))
model_base.add(Dense(256,activation='relu'))
model_base.add(Dense(128,activation='relu'))
model_base.add(Dense(64,activation='relu'))
model_base.add(Dense(10,activation='softmax'))

#compile
model_base.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

#train
history = model_base.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

#evaluate
loss,test_accuracy=model_base.evaluate(x_test,y_test)
print(f"test_accuracy:{test_accuracy}")

#*******************************************************************************************************************


#model 2 with l2 regularizers (1e-4)
#architecture
#*#
model_le4=Sequential()
model_le4.add(Flatten(input_shape=(32,32,3)))
model_le4.add(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(1e-4)))
model_le4.add(Dense(512,activation='relu',kernel_regularizer=regularizers.l2(1e-4)))
model_le4.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l2(1e-4)))
model_le4.add(Dense(128,activation='relu',kernel_regularizer=regularizers.l2(1e-4)))
model_le4.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(1e-4)))
model_le4.add(Dense(10,activation='softmax',kernel_regularizer=regularizers.l2(1e-4)))#we have 100 classes

model_le4.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
#train
history_le4=model_le4.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

#evaluate
loss,test_accuracy=model_le4.evaluate(x_test,y_test)
print(f"test_accuracy:{test_accuracy}")

#**************************************************************************************************

#model 3 with l2 regularizers le-2
model_le2 = Sequential()
model_le2.add(Flatten(input_shape=(32,32,3)))
model_le2.add(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(1e-2)))
model_le2.add(Dense(512,activation='relu',kernel_regularizer=regularizers.l2(1e-2)))
model_le2.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l2(1e-2)))
model_le2.add(Dense(128,activation='relu',kernel_regularizer=regularizers.l2(1e-2)))
model_le2.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(1e-2)))
model_le2.add(Dense(10,activation='softmax',kernel_regularizer=regularizers.l2(1e-2)))

#compile
model_le2.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

#train
history_le2 = model_le2.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

loss,test_accuracy = model_le2.evaluate(x_test,y_test)
print(f"test_accuracy:{test_accuracy}")

#visualization 
plt.plot(history.history['val_accuracy'],label='without regularizer')
plt.plot(history_le4.history['val_accuracy'],label='le4',color='red')
plt.plot(history_le2.history['val_accuracy'],label='le2',color='green')
plt.title("Validation accuracy")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()