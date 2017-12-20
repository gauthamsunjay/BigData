import tensorflow as tf
import time

ALPHA = -0.01
NUM_HOSTS = 5
NUM_FEATURES = 33762578
TOTAL_ITERATIONS = 20 * 1000 * 1000

# Trigger error calculation every x iterations
ERRORS_PER_X_ITERATIONS = 100 * 1000

# To calculate the errors over the entire test file
NUM_DATA_POINTS_PER_FILE = 2 * 1000 * 1000
ERRORS_FILE = "/home/ubuntu/assignment3/uploaded-scripts/sgd_errors.txt"

# We calculate the errors periodically over a small portion of the file as well
ERRORS_PER_X_ITERATIONS_SMALL = 100
NUM_DATA_POINTS_PER_FILE_SMALL = 10000
ERRORS_FILE_SMALL = "/home/ubuntu/assignment3/uploaded-scripts/sgd_errors_small.txt"

file_path = "/home/ubuntu/assignment3/uploaded-scripts/data/criteo-tfr/tfrecords%02d"
file_nums = [ range(0, 5), range(5, 10), range(10, 15), range(15, 20), [20, 21] ]
FILES = [
    map(lambda x: file_path % x, i) for i in file_nums
]
TEST_FILE = file_path % 22

print "calculating error over %d points every %d iterations" % (NUM_DATA_POINTS_PER_FILE_SMALL, ERRORS_PER_X_ITERATIONS_SMALL)
print "also calculating error over %d points every %d iterations" % (NUM_DATA_POINTS_PER_FILE, ERRORS_PER_X_ITERATIONS)

g = tf.Graph()
with g.as_default():

    # creating a model variable on task 0. This is a process running on node vm-19-1
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.ones(shape=[NUM_FEATURES], dtype=tf.float32), name='weights')

    sparse_gradients = []
    for i in range(NUM_HOSTS):
        with tf.device("/job:worker/task:%d" % i):
            filename_queue = tf.train.string_input_producer(FILES[i])

            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            features = tf.parse_single_example(serialized_example,
                features={
                    'label': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
                    'index' : tf.VarLenFeature(dtype=tf.int64),
                    'value' : tf.VarLenFeature(dtype=tf.float32),
                },
            )

            label = features['label']
            index = features['index']
            value = features['value']

            y = tf.cast(label, tf.float32)

            # getting only the required values from w that features would need to use
            sparse_w = tf.gather(w, index.values)
            x = value.values

            # ALPHA * y * (sigmoid(y * w.dot(x)) - 1) * x
            w_dot_x = tf.reduce_sum(tf.mul(sparse_w, x))
            partial_gradient = tf.mul(ALPHA, tf.mul(y, tf.mul(tf.sigmoid(tf.mul(y, w_dot_x)) - 1, x)))

            sparse_partial_gradient = tf.SparseTensor(shape=[NUM_FEATURES], indices=tf.transpose([index.values]), values=partial_gradient)
            sparse_gradients.append(sparse_partial_gradient)

    # we create an operator to aggregate the local gradients
    with tf.device("/job:worker/task:0"):

        sparse_addition = sparse_gradients[0]
        for sg in sparse_gradients[1:]:
            sparse_addition = tf.sparse_add(sparse_addition, sg)

        gradient_descent_op = tf.scatter_add(w, tf.reshape(sparse_addition.indices, [-1]), sparse_addition.values, use_locking=True)

    # Testing
    with tf.device("/job:worker/task:0"):
        filename_queue = tf.train.string_input_producer([TEST_FILE])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example,
            features={
                'label': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
                'index' : tf.VarLenFeature(dtype=tf.int64),
                'value' : tf.VarLenFeature(dtype=tf.float32),
            },
        )
        label = features['label']
        index = features['index']
        value = features['value']

        filtered_w = tf.gather(w, index.values)
        calculated_label = tf.reduce_sum(tf.mul(filtered_w, value.values))
        calculated_label = tf.cast(tf.sign(calculated_label), dtype=tf.int64)

        # comparing calculated_label with label
        is_error = tf.not_equal(label, calculated_label)

    with tf.Session("grpc://vm-19-1:2222") as sess:
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess=sess)

        def calc_error(n):
            print "calculating error over %d points" % n
            num_errors = 0
            for j in xrange(n):
                e = sess.run(is_error)
                if e[0]:
                    num_errors += 1

            error_percentage = (float(num_errors) / n) * 100.0
            print "error = ", error_percentage
            return error_percentage

        print "running 10 iterations to figure out average runtime..."
        runtimes = []
        for i in xrange(10):
            st = time.time()
            sess.run(gradient_descent_op)
            et = time.time()

            runtimes.append(et - st)

        avg_runtime = sum(runtimes) / len(runtimes)
        print "average runtime per training iteration = ", avg_runtime

        print "running the rest %d iterations" % (TOTAL_ITERATIONS - 10)
        for i in xrange(10, TOTAL_ITERATIONS):
            if i % 10 == 0:
                print "Iter #%05d" % i

            sess.run(gradient_descent_op)
            if i % ERRORS_PER_X_ITERATIONS == 0:
                error = calc_error(NUM_DATA_POINTS_PER_FILE)
                with open(ERRORS_FILE, 'a') as f:
                    f.write("%s after %s\n" % (error, i))
                continue

            if i % ERRORS_PER_X_ITERATIONS_SMALL == 0:
                error = calc_error(NUM_DATA_POINTS_PER_FILE_SMALL)
                with open(ERRORS_FILE_SMALL, 'a') as f:
                    f.write('%s after %s\n' % (error, i))
