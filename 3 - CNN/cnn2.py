import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import math
from utils.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training = 49000, num_validation = 1000, num_test = 10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test, mean_image

def setup_input():
    X = tf.placeholder(tf.float32, [None, 32, 32, 3], name = 'X')
    y = tf.placeholder(tf.int64, [None], name = 'y')
    is_training = tf.placeholder(tf.bool, name = 'is_training')
    return X, y, is_training

def setup_metrics(y, y_out):
    # Define loss function.
    total_loss = tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(y, 10), logits = y_out)
    mean_loss = tf.reduce_mean(total_loss)
    
    # Add top three predictions.
    prob = tf.nn.softmax(y_out)
    (guess_prob, guess_class) = tf.nn.top_k(prob, k = 3)
    
    # Compute number of correct predictions.
    is_correct = tf.equal(tf.argmax(y_out, 1), y)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    return mean_loss, accuracy, guess_prob, guess_class

def setup_scalar_summaries():
    tf.summary.scalar('mean_loss', mean_loss)
    tf.summary.scalar('accuracy', accuracy)
    all_summaries = tf.summary.merge_all()
    return all_summaries

def setup_optimizer(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Batch normalization in TensorFlow requires this extra dependency
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(loss)
    return train_step


# same function for both training and eval:
def run_iteration(session, X_data, y_data, training = None):
    # Set up variables we want to compute.
    variables = [mean_loss, accuracy, guess_prob, guess_class, all_summaries]
    if training != None:
        variables += [training]
        
    # Map inputs - placeholders to data.
    feed_dict = { X: X_data, y: y_data, is_training: training != None }
            
    # Compute variable values, and perform training step if required.
    values = session.run(variables, feed_dict = feed_dict)

    # Return loss value and number of correct predictions.
    return values[:-1] if training != None else values

# full 
def run_model(session, predict, loss_val, Xd, yd,
              epochs = 1, batch_size = 64, print_every = 100,
              training = None):
    
    dataset_size = Xd.shape[0]
    batches_in_epoch = int(math.ceil(dataset_size / batch_size))

    # Shuffle indices.
    train_indices = np.arange(dataset_size)
    np.random.shuffle(train_indices)

    # Count iterations since the beginning of training. 
    iter_cnt = 0

    for e in range(epochs):
        # Keep track of performance stats (loss and accuracy) in current epoch.
        total_correct = 0
        losses = []
        
        # Iterate over the dataset once.
        for i in range(batches_in_epoch):

            # Indices for current batch.
            start_idx = (i * batch_size) % dataset_size
            idx = train_indices[start_idx : (start_idx + batch_size)]
            
            # Get batch size (may not be equal to batch_size near the end of dataset).
            actual_batch_size = yd[idx].shape[0]
            
            loss, acc, _, _, summ = run_iteration(session, Xd[idx,:], yd[idx], training)

            # Update performance stats.
            losses.append(loss * actual_batch_size)
            total_correct += acc * actual_batch_size
            
            # Add summaries to event file.
            if (training is not None):
                writer.add_summary(summ, e * batches_in_epoch + i)
            
            # Print status.
            if (training is not None) and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2f}%"\
                      .format(iter_cnt, loss, acc * 100))
            iter_cnt += 1

        # Compute performance stats for current epoch.
        total_accuracy = total_correct / dataset_size
        total_loss = np.sum(losses) / dataset_size

        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.2f}%"\
              .format(total_loss, total_accuracy * 100, e + 1))
    return total_loss, total_correct

# ======== EXERCISE : CIFAR ============

# constructing an architecture given by a log file

def cifar10_net(X, y, is_training):
    # Convolution layer.

    print(X.shape)
    print(y.shape)

    conv1 = tf.layers.conv2d(inputs = X, filters = 32, kernel_size = [5, 5], padding = 'SAME', name = 'conv1')
    
    print(conv1.shape)

    # Pooling layer.
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [3, 3], strides = [2, 2], name = 'pool1')

    print(pool1.shape)

    after_relu = tf.nn.relu(pool1)

    conv2 = tf.layers.conv2d(inputs = after_relu, filters = 32, kernel_size = [5, 5], padding = 'SAME', activation = tf.nn.relu, name = 'conv2')
    

    print(conv2.shape)

    pool2 = tf.layers.average_pooling2d(inputs = conv2, pool_size = [3, 3], strides = [2, 2], name = 'pool2')
    
    print(pool2.shape)

    conv3 = tf.layers.conv2d(inputs = pool2, filters = 64, kernel_size = [5, 5], padding = 'SAME', activation = tf.nn.relu, name = 'conv3')
    pool3 = tf.layers.average_pooling2d(inputs = conv3, pool_size = [3, 3], strides = [2, 2], name = 'pool3')
    

    # First fully connected layer.
    pool3_flat = tf.reshape(pool3,[-1, 576])
    fc1 = tf.layers.dense(inputs = pool3_flat, units = 64, name = 'fc1')
    
    # Second fully connected layer.
    fc2 = tf.layers.dense(inputs = fc1, units = 10, name = 'fc2')

    return fc2

tf.reset_default_graph()
writer = tf.summary.FileWriter('./exercise')

X, y, is_training = setup_input()
y_out = cifar10_net(X, y, is_training)
mean_loss, accuracy, guess_prob, guess_class = setup_metrics(y, y_out)
all_summaries = setup_scalar_summaries()
train_step = setup_optimizer(mean_loss, 1e-3)

writer.add_graph(tf.get_default_graph())

# just in case
writer.flush()
writer.close()

X_train, y_train, X_val, y_val, X_test, y_test, mean_image = get_CIFAR10_data()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Training')
    run_model(sess, y_out, mean_loss, X_train, y_train, 8, 100, 100, train_step)
    print('Validation')
    run_model(sess, y_out, mean_loss, X_val, y_val, 1, 100)

'''
Saving and loading models
Saving is done using tf.train.Saver class:
save method saves both network definition and weights.
export_meta_graph method saves only network definition.
Loading is done in two stages:
tf.train.import_meta_graph function loads network definition, and returns a saver object that was used to save the model.
restore method of the returned saver object loads the weights.
Note that since weights are available only inside a session, save and restore methods above require a session object as a parameter.
'''

# slices?