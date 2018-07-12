"""
Day 4 part 2
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from dayFour1 import create_feature_set_and_labels 

# mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
train_x, train_y, test_x, test_y = create_feature_set_and_labels('pos.txt','neg.txt')

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100

X = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    
     
    layer1 = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    layer2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    layer3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    outputs = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                     'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data, layer1['weights']), layer1['biases'])
    l1 = tf.nn.relu(l1)
                                           
    l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
    l2 = tf.nn.relu(l2)                                       
                                           
    l3 = tf.add(tf.matmul(l2, layer3['weights']), layer3['biases'])
    l1 = tf.nn.relu(l3)
                                           
    output = tf.add(tf.matmul(l3, outputs['weights']), outputs['biases']) 
    return output 

def train_nn(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            
        correct = tf.equal(tf.nn.softmax_cross_entropy_with_logits_v2(prediction, 1), tf.nn.softmax_cross_entropy_with_logits_v2(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({X:test_x, y:test_y.labels}))
        
train_nn(X)