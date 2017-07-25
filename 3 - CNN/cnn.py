import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import math
from utils.vis_utils import visualize_grid
from utils.data_utils import load_CIFAR10


# don't show warnings:
# export TF_CPP_MIN_LOG_LEVEL=2

# === CIFAR10 data ===

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

def visualize_CIFAR10_sample(X, y, sample_count = 10, class_count = 10):
    set_size = X.shape[0]

    # Randomize dataset.
    data = np.ndarray([set_size, 2], dtype = np.int32)
    data[:, 0] = list(range(set_size))
    data[:, 1] = y
    data[:, :] = data[np.random.permutation(set_size)]
    
    # Select samples.
    selection = { i : [] for i in range(class_count) }
    count = 0
    for (ind, cls) in data:
        if len(selection[cls]) < sample_count:
            selection[cls] += [ind]
            count += 1
        if count == class_count * sample_count:
            break
    
    # Ensure that we found enough samples.
    assert count == class_count * sample_count
    
    # Flatten list.
    selection_flat = [item for cls in range(class_count) for item in selection[cls]]
    
    # Visualize samples.
    plt.figure(figsize = (12, 12))
    plt.imshow(visualize_grid((X[selection_flat, :, :, :] + np.reshape(mean_image, [1, 32, 32, 3]))))
    plt.axis("off")
    plt.show()

X_train, y_train, X_val, y_val, X_test, y_test, mean_image = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# show the grid
# visualize_CIFAR10_sample(X_train, y_train)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# === making a network ===

# clears any network present in memory
tf.reset_default_graph()

# log dir for TensorBoard for "event files"
log_dir = './logs_test/'
writer = tf.summary.FileWriter(log_dir)

# tf.placeholder: input variables
def setup_input():
    X = tf.placeholder(tf.float32, [None, 32, 32, 3], name = 'X')
    y = tf.placeholder(tf.int64, [None], name = 'y')
    is_training = tf.placeholder(tf.bool, name = 'is_training')
    return X, y, is_training

X, y, is_training = setup_input()

# add conv and pool layers
conv1 = tf.layers.conv2d(inputs = X, filters = 32, kernel_size = [7, 7], strides = 2, padding = 'SAME', activation=tf.nn.relu, name = 'conv1')
pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2, padding = 'SAME', name = 'pool1')

# write default graph to file: TensorBoard
# run "python -m tensorflow.tensorboard --logdir=./logs"
writer.add_graph(tf.get_default_graph())

# print all global vars (params, not placeholders=inputs) and find some 
print(tf.global_variables())

# print conv1 kernel and bias params
conv1_kernel = tf.get_default_graph().get_tensor_by_name('conv1/kernel:0')
conv1_bias = tf.get_default_graph().get_tensor_by_name('conv1/bias:0')
print("conv1 kernel shape: " + str(conv1_kernel.shape))
print("conv1 bias shape: " + str(conv1_bias.shape))

# inspect the shape of output tensors
print("conv1 output shape: " + str(conv1.shape))
print("pool1 output shape: " + str(pool1.shape))

# add a dense layer with relu

fc1_input_count = int(pool1.shape[1] * pool1.shape[2] * pool1.shape[3])

fc1_output_count = 1024

print("Fc1 input_cnt and output_cnt" + str([fc1_input_count, fc1_output_count]))

pool1_flat = tf.reshape(pool1, [-1, fc1_input_count])
fc1 = tf.layers.dense(inputs = pool1_flat, units = 1024, activation = tf.nn.relu, name = 'fc1')

# add another dense layer with identity and softmax afterwards
class_count = 10
fc2 = tf.layers.dense(inputs = fc1, units = class_count, name = 'fc2')

prob = tf.nn.softmax(fc2)
(guess_prob, guess_class) = tf.nn.top_k(prob, k = 3)

# visualizing parameters and activations
with tf.variable_scope('conv1_visualization'):
    # Normalize to [0 1].
    x_min = tf.reduce_min(conv1_kernel)
    x_max = tf.reduce_max(conv1_kernel)
    normalized = (conv1_kernel - x_min) / (x_max - x_min)

    # Transpose to [batch_size, height, width, channels] layout.
    transposed = tf.transpose(normalized, [3, 0, 1, 2])
    
    # Display random 5 filters from the 32 in conv1.
    # NOTE: this is just a regular TF node which is in this case an image
    # later we can add it to a writer to output this to tensorboard
    conv1_kernel_image = tf.summary.image('conv1/kernel', transposed, max_outputs = 3)
    
    # Do the same for output of conv1.
    sliced = tf.slice(conv1, [0, 0, 0, 0], [1, -1, -1, -1])
    x_min = tf.reduce_min(sliced)
    x_max = tf.reduce_max(sliced)
    normalized = (sliced - x_min) / (x_max - x_min)
    transposed = tf.transpose(normalized, [3, 1, 2, 0])
    conv1_image = tf.summary.image('conv1', transposed, max_outputs = 3)

# write default graph to file: TensorBoard
# run "python -m tensorflow.tensorboard --logdir=./logs"
writer.add_graph(tf.get_default_graph())

# forward pass

def choose_random_image(): # 1 x image
    index = np.random.randint(0, X_train.shape[0])
    image = X_train[[index], :, :, :]
    label = y_train[[index]]
    return index, image, label

random_index, random_image, random_label = choose_random_image()

# execute tf graph: tf.Session -> run

with tf.Session() as sess:
    with tf.device("/cpu:0") as dev: #"/cpu:0" or "/gpu:0"
        # Initialize weights for params/vars
        sess.run(tf.global_variables_initializer())

        # Map inputs (placeholders) to data.
        feed_dict = { X : random_image, y : random_label }

        # Set up variables we want to compute.
        # output and debug images!
        variables = [guess_prob, guess_class, conv1_kernel_image, conv1_image]

        # Perform forward pass.
        guess_prob_value, guess_class_value, conv1_kernel_value, conv1_value = sess.run(variables, feed_dict = feed_dict)

writer.add_summary(conv1_kernel_value)
writer.add_summary(conv1_value)

# visualize chosen image and predictions
def visualize_classification(image, guess_class, guess_prob):
    plt.imshow(image / 256.0) # FIX: imshow expects [0, 1]
    plt.axis("off")
    plt.show()
    for i in range(3):
        ind = guess_class[0, i]
        prob = guess_prob[0, i]
        print("Class: {0}\tProbability: {1:0.0f}%".format(class_names[ind], prob * 100))
    print("Ground truth: {0}".format(class_names[random_label[0]]))

# visualize random image, random image after norm, just the mean image
def vis_stuff():
    visualize_classification(random_image[0, :, :, :] + mean_image, guess_class_value, guess_prob_value)
    visualize_classification(random_image[0, :, :, :], guess_class_value, guess_prob_value)
    visualize_classification(mean_image, guess_class_value, guess_prob_value)

# vis_stuff()

# GETTING CLOSER TO TRAINING ==================================

# Setup metrics (e.g. loss and accuracy).

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

mean_loss, accuracy, guess_prob, guess_class = setup_metrics(y, fc2)

# show metrics

def setup_scalar_summaries():
    tf.summary.scalar('mean_loss', mean_loss)
    tf.summary.scalar('accuracy', accuracy)
    all_summaries = tf.summary.merge_all()
    return all_summaries

all_summaries = setup_scalar_summaries()

# setup optimizer

def setup_optimizer(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Batch normalization in TensorFlow requires this extra dependency
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(loss)
    return train_step

train_step = setup_optimizer(mean_loss, 5e-4)

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

# training for one epoch
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess, fc2, mean_loss, X_train, y_train, 1, 64, 100, train_step)
print('Validation')
_ = run_model(sess, fc2, mean_loss, X_val, y_val, 1, 64)

# visualize
random_index, random_image, random_label = choose_random_image()
_, _, guess_prob_value, guess_class_value, _ = run_iteration(sess, random_image, random_label)
visualize_classification(random_image[0, :, :, :] + mean_image, guess_class_value, guess_prob_value)

# just in case
# batch norm, new network!!!

def bn_model(X, y, is_training):
    # Convolution layer.
    conv1 = tf.layers.conv2d(inputs = X, filters = 32, kernel_size = [7, 7], strides = 2, padding = 'SAME', activation=tf.nn.relu, name = 'conv1')
    
    # Batch normalization layer.
    bn1 = tf.layers.batch_normalization(conv1, training = is_training)

    # Pooling layer.
    pool1 = tf.layers.max_pooling2d(inputs = bn1, pool_size = [2, 2], strides = 2, padding = 'SAME', name = 'pool1')

    # First fully connected layer.
    pool1_flat = tf.reshape(pool1,[-1, fc1_input_count])
    fc1 = tf.layers.dense(inputs = pool1_flat, units = 1024, activation = tf.nn.relu, name = 'fc1')
    
    # Second fully connected layer.
    fc2 = tf.layers.dense(inputs = fc1, units = class_count, name = 'fc2')

    return fc2

tf.reset_default_graph()
writer = tf.summary.FileWriter(log_dir)
X, y, is_training = setup_input()
y_out = bn_model(X, y, is_training)
mean_loss, accuracy, guess_prob, guess_class = setup_metrics(y, y_out)
all_summaries = setup_scalar_summaries()
train_step = setup_optimizer(mean_loss, 5e-4)

# train and validate
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Training')
    run_model(sess, y_out, mean_loss, X_train, y_train, 1, 64, 100, train_step)
    print('Validation')
    run_model(sess, y_out, mean_loss, X_val, y_val, 1, 64)

# just in case
writer.flush()
writer.close()

