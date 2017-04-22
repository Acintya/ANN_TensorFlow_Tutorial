import tensorflow as tf
import numpy as np

# define add_layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 1. set training dataset
# Make up some real data 
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 2. define placeholder for inputs to network  
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 3. define hidden layer and output layer
# add hidden layer: input xs£¬10 neuro
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# 4. define loss function
# the error between prediciton and real data    
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

# 5. choose optimizer for minimize loss value: GradientDescent
# and learing rate: 0.1       
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# important step: initialize all variables
init = tf.global_variables_initializer()
sess = tf.Session()
# run caculation with sess.run
sess.run(init)

# iteration for 1000 times£¬sess.run optimizer
for i in range(1000):
    # since training train_step and loss function are defined by placeholder £¬pass variables with feed
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # print result for every 50 steps to see the step improvement
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))