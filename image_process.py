import tensorflow as tf
sess = tf.Session()
import numpy as np


'''
>>>First we create our sample 2D image with numpy . This image will be a 4x4 pixel image. We will create it in four dimensions; the first and last
dimension will have a size of one. Note that some TensorFlow image functions will operate on four-dimensional images. Those four
dimensions are image number, height, width, and channel, and to make it one image with one channel, we set two of the dimensions to 1, as
follows:
'''

x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)
print(x_val)
x_data = tf.placeholder(tf.float32, shape=x_shape)
'''
>>>To create a moving window average across our 4x4 image, we will use a built-in function that will convolute a constant across a window of
the shape 2x2 . This function is quite common to use in image processing and in TensorFlow, the function we will use is conv2d() . This
function takes a piecewise product of the window and a filter we specify. We must also specify a stride for the moving window in both
directions. Here we will compute four moving window averages, the top left, top right, bottomleft, and bottomright four pixels. We do this
by creating a 2x2 window and having strides of length 2 in each direction. To take the average, we will convolute the 2x2 window with a
constant of 0.25 ., as follows:

'''
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]
mov_avg_layer= tf.nn.conv2d(x_data, my_filter, my_strides,
padding='SAME''', name='Moving_Avg_Window')

r"""Computes a 2-D convolution given 4-D `input` and `filter` tensors.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`, this op
performs the following:

1. Flattens the filter to a 2-D matrix with shape
   `[filter_height * filter_width * in_channels, output_channels]`.
2. Extracts image patches from the input tensor to form a *virtual*
   tensor of shape `[batch, out_height, out_width,
   filter_height * filter_width * in_channels]`.
3. For each patch, right-multiplies the filter matrix and the image patch
   vector.

In detail, with the default NHWC format,

    output[b, i, j, k] =
        sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                        filter[di, dj, q, k]

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

Args:
  input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
    A 4-D tensor. The dimension order is interpreted according to the value
    of `data_format`, see below for details.
  filter: A `Tensor`. Must have the same type as `input`.
    A 4-D tensor of shape
    `[filter_height, filter_width, in_channels, out_channels]`
  strides: A list of `ints`.
    1-D tensor of length 4.  The stride of the sliding window for each
    dimension of `input`. The dimension order is determined by the value of
    `data_format`, see below for details.
  padding: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.
  use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
  data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
    Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, height, width, channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, channels, height, width].
  dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    1-D tensor of length 4.  The dilation factor for each dimension of
    `input`. If set to k > 1, there will be k-1 skipped cells between each
    filter element on that dimension. The dimension order is determined by the
    value of `data_format`, see above for details. Dilations in the batch and
    depth dimensions must be 1.
  name: A name for the operation (optional).

Returns:
  A `Tensor`. Has the same type as `input`.
"""

def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b) # Ax + b
    return(tf.sigmoid(temp))


with tf.name_scope('Custom_Layer') as scope:
     custom_layer1 = custom_layer(mov_avg_layer)
     print(sess.run(custom_layer1, feed_dict={x_data: x_val}))
