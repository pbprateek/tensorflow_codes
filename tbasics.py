# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 00:03:18 2018

@author: prateek
"""

import tensorflow as tf
x1=tf.constant(5)
x2=tf.constant(6)
#result=x1*x2
result=tf.multiply(x1,x2)
print(result)

with tf.Session() as sess:
    output=sess.run(result)
    print(output)
    
    
print(sess.run(result)) #error bcz u are out of session

