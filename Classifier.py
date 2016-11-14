import tensorflow as tf

EXAMPLES = 1000 # HOW BIG THE DATASET WILL BE
display_step = 1

x = tf.placeholder("float", [None, 38]) # Inputs
y = tf.placeholder("float", [None, 2])  # Classes

weights = {
    'h1': tf.Variable(tf.random_normal([38, 256])),
    'h2': tf.Variable(tf.random_normal([256, 256])),
    'out': tf.Variable(tf.random_normal([256, 2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([256])),
    'b2': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([2]))
}

layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
model = tf.matmul(layer2, weights['out']) + biases['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(15):
        avg_cost = 0
        total_batch = int(EXAMPLES/100)
        #for i in range(total_batch):
            #batchX, batchY = Get the landmarks from the images.
            #_, c = sess.run([optimizer, cost], feed_dict={x: batchX, y: batchY})
            #avg_cost += c / total_batch
        if epoch % 1 == 0:
            print "Epoch", '%04d' % (epoch+1), "cost = ", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"
    correctPrediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))
    #print "Accuracy:", accuracy.eval({x: TEST_IMAGES, y: TEST_LABELS})
