"""
This is the file for 100days of ML
I alredy had basic knowledge in ML, In this series of 100 days I aim to master 
some deep learning technique, specicifically here in day one I am following 
sentdex's tutorial Tensorflow Basics-Deep learning with NNs
"""

"""
Day one
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

X = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    
     
    layer1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
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
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({X:mnist.test.images, y:mnist.test.labels}))
        
train_nn(X)