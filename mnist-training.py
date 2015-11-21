import input_data
import tensorflow as tf

# MNIST DATA
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x is not a specific value. It is a placeholder, a value that we'll input
# when we ask TensorFlow to run a computation
# None means a dimension can be any length
x = tf.placeholder("float", [None, 784])

# Variables
# : a modifiable tensor that lives in TensorFlow's graph of interacting 
# operations. Can be used and even modified by the computation.
# One generally has the model parameters be Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# dimensions?
# x * W + b
# x is [None, 784]
# W is [784, 10]
# b is [10, 1]


y = tf.nn.softmax(tf.matmul(x, W) + b) # size is [10,1] (this is our prediction)

# implementing cross entropy (our cost function)
y_ = tf.placeholder("float", [None, 10]) # this placeholder for real values
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))


# minimize cross_entropy using gradient descent
# learning rate (alpha) is 0.01
# what TF does here (behind the scenes) is adds new operations to your graph which
# implements back-propagation and gradient descent
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initialize the Variables we created
init = tf.initialize_all_variables()

# launch the model in a Session
sess = tf.Session()
sess.run(init)

# train that model (run 1000 times)
for i in range(1000):
    # get batch of 100 random data points from training set
    # use small batches of data at a time to slowly train our model, so that
    # it is not super expensive
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})


# Evaluating our model
# tf.argmax gives us the index of the highest entry in a tensor along some axis
# example: tf.argmax(y, 1) is the label our model thinks is most likely for each input
# tf.argmax(y_, 1) is the correct label

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # list of booleans

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})
