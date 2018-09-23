# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 00:23:15 2018

@author: prateek
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


#onehot is same as onehotencoder from sklearn
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

n_nodes_hl1=500
n_nodes_hl2=500
n_nodes_hl3=500

n_classes=10
batch_size=100 #it will go through 100 images and manipulate weight

#28*28 is column
#A placeholder is simply a variable that we will assign data to at a later date
#The first dimension of the placeholder is None, meaning
# we can have any number of rows. The second dimension is fixed at 784, meaning
# each row needs to have 784 columns of data.
x=tf.placeholder('float',shape=[None, 784],name='x')
y=tf.placeholder('float')

#The difference is that with tf.Variable you have to provide an initial value when
# you declare it. With tf.placeholder you don't have to provide an initial value and
# you can specify it at run time with the feed_dict argument inside Session.run

#In short, you use tf.Variable for trainable variables such as weights
#tf.placeholder is used to feed actual training examples
def neural_network_model(data):
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                    'biases':tf.Variable(tf.zeros([n_nodes_hl1]))}
    
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                    'biases':tf.Variable(tf.zeros([n_nodes_hl2]))}
    
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                    'biases':tf.Variable(tf.zeros([n_nodes_hl3]))}
    
    outer_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                    'biases':tf.Variable(tf.zeros([n_classes]))}
    
    #tf.matmul is matrix multiplication
    #tf.add is equivalent to x+y and supports broadcasting,tf.add_n dosen't
    #support broadcasting
    
    l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1=tf.nn.relu(l1)
    
    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2=tf.nn.relu(l2)
    
    l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3=tf.nn.relu(l3)
    
    output=tf.matmul(l3,outer_layer['weights']) + outer_layer['biases']
    
    return output


def train_neural_network(x):
    prediction=neural_network_model(x)
    #Computes softmax cross entropy between logits and labels.
    #cross-entropy loss is what we use to measure the error at a softmax layer,see it like a logistic cost 
    #function for now(will update it when it gets clear to me.)
    #and reduce_mean to add up all the cost and divide it by m.
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )
    
    
    #use optimizer of ur choice
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    
    #basically no of iterations
    hm_epochs =5
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # _(underscore) -they are just variables we don't care about
        
        for epoch in range(hm_epochs):
            
            epoch_loss=0
            #mnist.train.num_examples/batch_size - this is basically no of iterations required to go through all the data.
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y=mnist.train.next_batch(batch_size)
                
                
                #sess.run(fetches,feed_dict=None,options=None,run_metadata=None)
                #fetches argument may be a single graph element, or an arbitrarily
                #nested list, tuple, namedtuple, dict, or OrderedDict
                #The value returned by run() has the same shape as the fetches argument.
                
                #Each value in feed_dict must be convertible to a numpy array of the dtype of the corresponding key.
                #
                _,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                #we are summing up the cost from each batch and printing it per interation or epoch
                #it should reduce each epoch
                epoch_loss=epoch_loss+c
            
            print('Epoch',epoch,'compleated out of',hm_epochs, 'loss:',epoch_loss)
        
        
        #tf.equals- Returns the truth value of (x == y) 
        #tf.argmax- it returns the maximum value across that axix,
        #so basically prediction will be of [m,10] and y will also be of size [m,10]
        #Something like 1 0 0 0 0 0 0 0 0 0 =0 ,0 0 0 0 0 0 0 1 0 0 =7
        #so if the index is same means it is correct prediction
        correct=tf.equal(tf.argmax(prediction,axis=1),tf.argmax(y,axis=1))
        
        #tf.cast - it will convert True to 1 and false to 0,so reduce_mean will tell us accuracy
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        
        #you can use sess.run() to fetch the values of many tensors in the same step.
        #If t is a Tensor object, t.eval() is shorthand for sess.run(t),where sess is the current default session
        print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
        
train_neural_network(x)    
    
    
    
    









