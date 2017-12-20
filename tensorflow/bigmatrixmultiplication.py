import tensorflow as tf

N = 100 * 1000 # 0 # dimension of the matrix
d = 100 # number of splits along one dimension. Thus, we will have 100 blocks
M = int(N / d)

def get_block_name(i, j):
    return "sub-matrix-"+str(i)+"-"+str(j)

def get_intermediate_trace_name(i, j):
    return "inter-"+str(i)+"-"+str(j)

def get_task_num_for_block(i, j, num_workers, cache={}):
    if (i,j) in cache or (j,i) in cache:
        return cache[(i,j)]

    task_num = (i + j) % num_workers
    cache[(i,j)] = task_num
    cache[(j,i)] = task_num
    return task_num

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(1024)

    matrices = {}
    for i in range(0, d):
        for j in range(0, d):
            task_num = get_task_num_for_block(i, j, 5) # 5 workers
            with tf.device("/job:worker/task:%d" %  task_num):
                matrix_name = get_block_name(i, j)
                matrices[matrix_name] = tf.random_uniform([M, M], name=matrix_name)

    intermediate_traces = {}
    for i in range(0, d):
        for j in range(0, d):
            task_num = get_task_num_for_block(i, j, 5)
            with tf.device("/job:worker/task:%d" % task_num):
                A = matrices[get_block_name(i, j)]
                B = matrices[get_block_name(j, i)]
                intermediate_traces[get_intermediate_trace_name(i, j)] = tf.trace(tf.matmul(A, B))

    with tf.device("/job:worker/task:0"):
        retval = tf.add_n(intermediate_traces.values())

    config = tf.ConfigProto(log_device_placement=True)
    with tf.Session("grpc://vm-19-2:2222", config=config) as sess:
        result = sess.run(retval)
        sess.close()
        print "RESULT: trace = %s" % result
