import tensorflow as tf
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)
x = [[[1,0,0,0]]]

nh = 2

cell = tf.contrib.rnn.BasicRNNCell(num_units=nh)
print(cell.output_size, cell.state_size)
x_data = np.asarray(x , dtype = np.float32)

output , state = tf.nn.dynamic_rnn(cell , x_data , dtype = tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("start")
print(output.eval(session = sess))
print("done")