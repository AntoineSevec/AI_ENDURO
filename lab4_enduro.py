#TEST
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:39:44 2022

@author: simpletree
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:12:20 2022

@author: simpletree
"""
#import everything but some lib are not used

import gym
import imageio
import numpy as np
from gym.utils.play import play
import pygame
import matplotlib
import argparse
from gym import logger
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#%% record training data
def mycallback(obs_t, obs_tp1, action, rew, done, info):
    print("action = ", action, " reward = ", rew, "done = ", done)
    list2 = [action]
    vector1 = np.array(list2)
    #imageio.imwrite('outfile.png', (obs_t-obs_tp1))#[55:155:2, 20:160:2, 1])
    with open('X_enduro.txt', 'a') as outfileX:
       np.savetxt(outfileX, delimiter=',', X=(obs_t)[55:155:2, 20:160:2, 1], fmt='%d')
    with open('Y_enduro.txt', 'a') as outfileY:
        np.savetxt(outfileY, delimiter='', X=vector1, fmt='%d')
    #return [action,]

#plotter = gym.utils.play.PlayPlot(mycallback, 30 * 5, ["action"]) #plot in real-time

env = gym.make("Enduro-v0")

play(env, zoom=4, fps=30, callback=mycallback)
env.close()

#%% load the dataset
#   X,Y is the training and label dataset
size_of_input = [70,50]
size_of_label = 9

X = np.loadtxt('X_enduro.txt',delimiter=',')
Y_data = np.loadtxt('Y_enduro.txt',delimiter=',')
X = X.reshape((Y_data.size,size_of_input[0],size_of_input[1],1))

Y=np.empty([0,size_of_label])   # size of Y is (number of sample data,9), each column represents the action
for i in Y_data:  #label encoding
    for j in range(size_of_label):
        if i == j:
            arr = np.zeros(size_of_label)
            arr[j]=1
            Y=np.append(Y,arr)
            
Y=Y.reshape(Y_data.size,size_of_label)

# split the data
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=np.random)

#%% build the model
model = Sequential()

model.add(Input(shape=(size_of_input[0],size_of_input[1],1)))

model.add(Conv2D(filters=3, kernel_size=(6, 6), activation='relu',strides=(2, 2)))
#model.add(Conv2D(filters=3, kernel_size=(6, 6), activation='relu',strides=(1, 1)))
model.add(tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), strides=(3,3), padding='same'))
#model.add(Conv2D(filters=1, kernel_size=(2, 2), activation='softmax',strides=(2, 2), padding='same'))
model.add(Flatten())
#model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=10, activation='sigmoid'))
model.add(Dense(units=size_of_label, activation='sigmoid'))
opt = tf.keras.optimizers.Adam(learning_rate=1*10e-4)  #change learning rate here
model.summary()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#%%train the model

history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
zz=model.predict(x_test) #see the prediction
#%% plot the result
'''
import matplotlib.pyplot as plt

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.show()
'''
#%%  model evaluation
'''
model.evaluate(x_test, y_test, verbose=2)
pred_test = model.predict(x_test)
print(tf.math.confusion_matrix(y_test.argmax(axis=1), pred_test.argmax(axis=1)))

# you can save the model here
#model.save('pong_game.h5')
#%%
# use model.predict() to get the prediction from input
#zz=model.predict(x_test)
'''
#model.save('enduro_game1.h5')

#model=tf.keras.models.load_model('enduro_game1.h5')
#%% run the agent
import gym
import pygame
env = gym.make('Enduro-v0')
fps=30 #30 frame per second is default ratio
max_iteration = 300
size_of_input = [70,50]
zoom = 4
#size_of_label = 9

def action_mapping(array):
    #i = np.nonzero(array)[0][0] #return the index of the action
    i = np.argmax(array)
    return i

obs = env.reset()[55:155:2, 20:160:2, 1].reshape(1,size_of_input[0],size_of_input[1],1)
action = np.array([1]) # first action = 0
#size_of_input[0],size_of_input[1]

for t in range(max_iteration): #the part of prediction is slow due to the tensor operation
#while(1):
    env.render()
    if action[t] !=0:
        print("action:",action[t])    
    obs,rew,d,inf=env.step(action[t]) # take a predicted action
    obs = obs[55:155:2, 20:160:2, 1].reshape(1,size_of_input[0],size_of_input[1],1) #reshape to fit in the input layer of model
    action = np.append(action,action_mapping(model.predict(obs)))
    if rew != 0:
        print("reward: ", rew)
    #pygame.time.Clock().tick(fps)
env.close()

#%% 
env.reset()
for t in range(max_iteration): #the part of prediction is slow due to the tensor operation
#while(1):
    env.render()
    obs,rew,d,inf=env.step(action[t]) # take a predicted action
    if rew != 0:
        print("reward: ", rew)
    pygame.time.Clock().tick(fps)
env.close()
