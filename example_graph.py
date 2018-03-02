import tensorflow as tf
sess = tf.Session()
import numpy as np



x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)
my_product = tf.multiply(x_data, m_const)
for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data: x_val}))
'''
>>> 
3.0
9.0
15.0
21.0
27.0
'''
my_array = np.array([[1., 3., 5., 7., 9.],
[-2., 0., 2., 4., 6.],
[-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
x_data = tf.placeholder(tf.float32, shape=(3, 5))

m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

print(x_vals)
'''
>>>[[[ 1.  3.  5.  7.  9.]
  [-2.  0.  2.  4.  6.]
  [-6. -3.  0.  3.  6.]]

 [[ 2.  4.  6.  8. 10.]
  [-1.  1.  3.  5.  7.]
  [-5. -2.  1.  4.  7.]]]

'''
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))
'''
>>>
[[102.]
 [ 66.]
 [ 58.]]
[[114.]
 [ 78.]
 [ 70.]]
'''