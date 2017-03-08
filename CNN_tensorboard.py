import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3  
x_data = np.random.rand(1000, 1).astype(np.float32)
y_data = tf.sin(x_data) * tf.cos(x_data) + tf.random_uniform([1000, 1], -0.1, 0.1)

# graph
X = tf.placeholder(tf.float32, [None, 1], name='X-input')
Y = tf.placeholder(tf.float32, [None, 1], name='Y-input')

W1 = tf.Variable(tf.random_uniform([1, 5], -1.0, 1.0), name='weight1')
W2 = tf.Variable(tf.random_uniform([5, 2], -1.0, 1.0), name='weight2')
W3 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name='weight3')

b1 = tf.Variable(tf.zeros([5]), name='bias1')
b2 = tf.Variable(tf.zeros([2]), name='bias2')
b3 = tf.Variable(tf.zeros([1]), name='bias3')

with tf.name_scope('layer2') as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope('layer3') as scope:
    L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)

with tf.name_scope('layer4') as scope:
    hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis))
    cost_summery = tf.scalar_summary("cost", cost)

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(cost)

# the summery
w1_hist = tf.histogram_summary("weight1", W1)
w2_hist = tf.histogram_summary("weight2", W2)
b1_hist = tf.histogram_summary("bisa1", b1)
b2_hist = tf.histogram_summary("bisa2", b2)
y_hist = tf.histogram_summary("y", Y)

init = tf.initialize_all_variables()

# run
with tf.Session() as sess:
    sess.run(init)
    # the workers who translate data to TensorBoard
    merged = tf.merge_all_summaries()  # collect the tf.xxxxx_summary
    writer = tf.train.SummaryWriter('keep', sess.graph)
    # maybe many writers to show different curvs in the same figure
    for step in range(20000):
        summary, _ = sess.run([merged, train], feed_dict={X: x_data, Y: y_data.eval()})
        writer.add_summary(summary, step)
        if step % 10 == 0:
            print('step %s' % (step)) 