import pickle
import tensorflow as tf
import os
#从本地提取数据
with open('model/params.p','rb') as f:
    title_count, title_set, genres2int, features, \
    targets_values, ratings, users, movies, data,\
    movies_orig, users_orig = pickle.load(f)

# features特征 UserID', 'MovieID','Gender','Age','JobID','Title', 'Genres'
#编码实现
#嵌入矩阵的维度
embed_dim = 32
#下面之所以要+1是因为编号和实际数量之间是差1的
#用户ID个数
uid_max = max(features.take(0,1))+1
#性别个数
gender_max = max(features.take(2,1))+1
#年龄类别个数
age_max = max(features.take(3,1))+1
#职业个数
job_max = max(features.take(4,1))+1
#电影ID个数
movie_id_max = max(features.take(1,1))+1
#电影类型个数 有一个<PAD>
print(genres2int)
movies_genres_max = max(genres2int.values())+1
print(movies_genres_max)
#电影名单词个数
movie_title_max = len(title_set)
#对电影类型嵌入向量做加和操作标志
combiner = 'sum'
#电影名长度，做词嵌入要求输入的维度是固定的，这里为15
#长度不够用空白符填充，太长则进行截断
sentence_size = title_count # 15
#文本卷积滑动窗口，分别滑动2,3,4,5个单词
window_sizes = {2,3,4,5}
#文本卷积核数量
filter_num = 8

#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]:i for i, val in enumerate(movies.values)}

#超参数
num_epochs = 5
batch_size = 256
dropout_keep = 0.5
learning_rate = 0.0001
# 显示每n个批次的统计数据
show_every_n_batches = 50
save_dir = './save'

#输入
#定义输入的占位符
def get_inputs():
    uid = tf.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")

    movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
    #电影种类中要去除<PAD>，所以-1
    movie_genres =tf.placeholder(tf.int32,[None,movies_genres_max-1],name='movie_genres')
    movie_title = tf.placeholder(tf.int32,[None,15],name='movie_title')
    targets = tf.placeholder(tf.int32, [None, 1], name="targets")
    learningRate = tf.placeholder(tf.float32,name='learningRate')
    dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')
    return uid,user_gender,user_age,user_job,movie_id,movie_genres,\
            movie_title,targets,learningRate,dropout_keep_prob

#构建神经网络
#定义user的嵌入层
def get_user_embedding(uid,user_gender,user_age,user_job):
    with tf.name_scope('user_embedding'):
        #先初始化一个非常大的用户特征
        #tf.random_matrix 的第二个参数是初始化的最小值
        #这里是-1，第三个参数是初始化的最大值，为1
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max,embed_dim],-1,1),\
                                       name='uid_embed_matrix')
        #根据指定用户ID找到他对应的嵌入层
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix,uid,\
                                        name='uid_embed_layer')
        #性别的特征维度设置为16
        gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max,embed_dim//2],-1,1),\
                                        name = 'gender_embed_matrix')
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix,user_gender,
                                         name='gender_embed_layer')
        #年龄特征维度设置为16
        age_embed_matrix = tf.Variable(tf.random_uniform([age_max,embed_dim//2],-1,1),\
                                       name='age_embed_matrix')
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix,user_age,\
                                        name='age_embed_layer')
        #职业的特征维度设置为16
        job_embed_matrix = tf.Variable(tf.random_uniform([job_max,embed_dim//2],-1,1),\
                                       name='job_embed_matrix')
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix,user_job,\
                                                 name='job_embed_layer')
        #返回产生的用户数据
        return uid_embed_layer,gender_embed_layer,age_embed_layer,job_embed_layer
#将user的嵌入矩阵一起全连接成user的特征
def get_user_feature_layer(uid_embed_layer,gender_embed_layer,age_embed_layer,job_embed_layer):
    with tf.name_scope('user_fc'):
        #第一层全连接
        #tf.layers.dense 的第一个参数是输入，第二个参数是层的单元的数量
        uid_fc_layer = tf.layers.dense(uid_embed_layer,embed_dim,\
                                name='uid_fc_layer',activation=tf.nn.relu)
        gender_fc_layer = tf.layers.dense(gender_embed_layer,embed_dim,\
                                name='gender_fc_layer',activation=tf.nn.relu)
        age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim,\
                                name="age_fc_layer", activation=tf.nn.relu)
        job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, \
                                name="job_fc_layer", activation=tf.nn.relu)
        #第二层全连接
        #将上面的每个分段组成一个完整的全连接层
        user_combine_layer = tf.concat([uid_fc_layer,gender_fc_layer,age_fc_layer,\
                                        job_fc_layer],2)#(?,1,128)
        #验证上面的tensorflow是否是128维
        # print(user_combine_layer.shape)
        # tf.contrib.layers.fully_connected 的第一个参数是输入，第二个参数是输出
        # 这里的输入是user_combine_layer，输出是200，是指每个用户有200个特征
        # 相当于是一个200个分类的问题，每个分类的可能性都会输出，在这里指的就是每个特征的可能性
        user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer,200,tf.tanh)#(?,1,200)
        user_combine_layer_flat = tf.reshape(user_combine_layer,[-1,200])
        # print(user_combine_layer_flat.shape)
        return user_combine_layer,user_combine_layer_flat

#定义movie_ID的嵌入矩阵
def get_movie_ID_embed_layer(movie_id):
    with tf.name_scope('movie_embedding'):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max,embed_dim],-1,1),\
                                            name='movie_id_embed_matrix')
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix,movie_id,\
                                            name='movie_id_embed_layer')
    return movie_id_embed_layer

#对电影类型的多个嵌入向量做加和
def get_movie_genres_layers(movie_genres):
    with tf.name_scope('movie_genres_layers'):
        movies_genres_embed_matrix = tf.Variable(tf.random_uniform([movies_genres_max,embed_dim],-1,1),\
                                            name='movies_genres_embed_matrix')
        movies_genres_embed_layers = tf.nn.embedding_lookup(movies_genres_embed_matrix,movie_genres,\
                                            name='movies_genres_embed_layers')
        # print(movies_genres_embed_layers.shape)
        if combiner=='sum':
            #keep_dims:表示是否保留原始数据的维度，False相当于执行完后原始数据就会少一个维度
            movies_genres_embed_layers = tf.reduce_sum(movies_genres_embed_layers,axis=1,keep_dims=True)
    return movies_genres_embed_layers


#movie_title的文本卷积网络实现
def get_movie_cnn_layer(movie_title,dropout_keep_prob):
    #从嵌入层中得到电影名岁月的各个单词的嵌入向量
    with tf.name_scope('movie_embedding'):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max,embed_dim],-1,1),\
                                        name='movie_title_embed_matrix')
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix,movie_title,\
                                        name='movie_title_embed_layer')
        # print(movie_title_embed_layer.shape)#(?,15,32)
        # 为 movie_title_embed_layer 增加一个维度
        # 在这里是添加到最后一个维度，最后一个维度是channel  (?,15,32) ==> (?,15,32,1)
        # 所以这里的channel数量是1个
        # 所以这里的处理方式和图片是一样的,4维
        #expand_dims  在第axis位置增加一个维度,
        # 尺寸索引轴从零开始; 如果您指定轴的负数，则从最后向后计数。
        movie_title_embed_layer_expend = tf.expand_dims(movie_title_embed_layer,-1)
        # print(movie_title_embed_layer_expend.shape)#(?,15,32,1)
    #对文本嵌入层使用不同的尺寸的卷积核做卷积和最大池化
    pool_layer_list = []
    # 文本卷积滑动窗口，分别滑动2,3,4,5个单词
    # window_sizes = {2, 3, 4, 5}
    # 文本卷积核数量
    # filter_num = 8
    for window_size in window_sizes:
        with tf.name_scope('movie_txt_conv_maxpool_{}'.format(window_size)):
            #[window_size,embed_dim,1,filter_num]表示输入的 channel 的个数是1，
            # 输出的 channel 的个数是 filter_num               2,32,1,8
            filter_weights = tf.Variable(tf.truncated_normal([window_size,embed_dim,1,filter_num],\
                                                    stddev=0.1,name='filter_weights'))
            filter_bias = tf.Variable(tf.constant(0.1,shape=[filter_num],name='filter_bias'))
            # conv2d是指用到的卷积核的大小是[filter_height * filter_width * in_channels, output_channels]
            #这里的卷积核会想两个维度的方向滑动
            # conv1d是将卷积核向一个维度的方向滑动。这就是conv1d和conv2d的区别
            #strides设置要求是第一个和最后一个数字是1，四个数字的顺序要求默认是NHWC
            #也就是[batch, height, width, channels]
            #padding设置好为VALID其实就是不PAD，为SAME是让输入和输出的维度一样
            conv_layer = tf.nn.conv2d(movie_title_embed_layer_expend,filter_weights,[1,1,1,1],\
                                      padding='VALID',name='conv_layer')
            # tf.nn.bias_add()价格偏差filter_bias加到conv_layer上
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer,filter_bias))
           #这边的池化是将上面每个卷积核的卷积结果转换为一个元素
#=============================
            #value, ksize, strides, padding
            maxpool_layer = tf.nn.max_pool(relu_layer,[1,sentence_size-window_size+1,1,1],\
                                    [1,1,1,1],padding='VALID',name='maxpool_layer')
            # print(maxpool_layer.shape)#(?,1,1,8)
            pool_layer_list.append(maxpool_layer)
    with tf.name_scope('pool_dropout'):
        # 这里最终的结果是这样的，
        # 假设卷积核的窗口是 2，卷积核的数量是 8
        # 那么通过上面的池化操作之后，生成的池化的结果是一个具有 8 个元素的向量
        # 每种窗口大小的卷积核经过池化后都会生成这样一个具有 8 个元素的向量
        # 所以最终生成的是一个 8 维的二维矩阵，它的另一个维度就是不同的窗口的数量
        # 在这里就是 2,3,4,5，那么最终就是一个 8*4 的矩阵，
        pool_layer = tf.concat(pool_layer_list,3,name='pool_layer')
        # print(pool_layer.shape)#（?,1,1,8）
        max_num = len(window_sizes)*filter_num#为了让max_num = 4 *8 =32
        # print(max_num)
        #将这个8*4的二维矩阵平铺长一个具有32个元素的一维矩阵
        pool_layer_flat = tf.reshape(pool_layer,[-1,1,max_num],name='pool_layer_flat')#(?,1,32)
        dropout_layer = tf.nn.dropout(pool_layer_flat,dropout_keep_prob,name='dropout_layer')
    return pool_layer_flat,dropout_layer#(?,1,32)

# print(pool_layer_flat.shape)
# print(dropout_layrer.shape)
#将movie的各个层一起做全连接层
def get_movie_feature_layer(movie_id_embed_layer,movies_genres_embed_layers,dropout_layer):
    with tf.name_scope('movie_fc'):
        #第一层全连接
        movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer,embed_dim,\
                                name='movie_id_fc_layer',activation=tf.nn.relu)
        # print(movie_id_embed_layer.shape)#(?,1,32)
        movies_genres_fc_layer = tf.layers.dense(movies_genres_embed_layers,embed_dim,\
                                name='movies_genres_fc_layer',activation=tf.nn.relu)
        # print(movies_genres_embed_layers.shape)#(?,1,32)
        # print(dropout_layer.shape)
        #第二层全连接
        movie_combine_layer = tf.concat([movie_id_fc_layer,dropout_layer,movies_genres_fc_layer],2)#(?,1,96)
        # print(movie_combine_layer.shape)
        movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer,200,tf.tanh)#(?,1,200)
        movie_combine_layer_flat = tf.reshape(movie_combine_layer,[-1,200])
        # print(movie_combine_layer_flat.shape)
    return movie_combine_layer, movie_combine_layer_flat

#构建计算图
# reset_default_graph 操作应该在 tensorflow 的其他所有操作之前进行，否则将会出现不可知的问题
# tensorflow 中的 graph 包含的是一系列的操作和使用到这些操作的 tensor
tf.reset_default_graph()
train_graph = tf.Graph()
# Graph 只是当前线程的属性，如果想要在其他的线程使用这个 Graph，那么就要像下面这样指定
# 下面的 with 语句将 train_graph 设置为当前线程的默认 graph
with train_graph.as_default():
    #获取输入占位符
    uid, user_gender, user_age, user_job, movie_id, movie_genres, \
    movie_title, targets, learningRate, dropout_keep_prob = get_inputs()
    #获取user的4个嵌入向量
    uid_embed_layer, gender_embed_layer, age_embed_layer, \
    job_embed_layer = get_user_embedding(uid, user_gender, user_age, user_job)
    #得到用户特征
    user_combine_layer, user_combine_layer_flat = \
        get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer)
    #获取电影ID的嵌入向量
    movie_id_embed_layer = get_movie_ID_embed_layer(movie_id)
    #获取电影类型的嵌入向量
    movies_genres_embed_layers = get_movie_genres_layers(movie_genres)
    #获取电影名的特征向量
    pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_title,dropout_keep_prob)
    #得到电影特征
    movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(\
                movie_id_embed_layer,movies_genres_embed_layers, dropout_layer)
    # 计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
    # tensorflow 的 name_scope 指定了 tensor 范围，方便我们后面调用，通过指定 name_scope 来调用其中的 tensor
    with tf.name_scope("inference"):
        #直接将用户的特征矩阵和电影特征矩阵相乘得到得分，最后要做的就是对着个得分进行回归
        inference = tf.reduce_sum(user_combine_layer_flat*movie_combine_layer_flat,axis=1)
        inference = tf.expand_dims(inference,axis=1)
    with tf.name_scope('loss'):
        #MSE损失，将计算值回归到评分
        cost = tf.losses.mean_squared_error(targets,inference)
        #将每个维度的cost相加，计算平均值
        loss = tf.reduce_mean(cost)
        #优化损失
        # train_op = tf.train.AdamOptimizer(learningRate).minimize(loss)  # cost
        # 在为 tensorflow 设置 name 参数的时候，是为了能在 graph 中看到什么变量进行了什么操作
    global_step = tf.Variable(0,name='global_step',trainable=False)
    optimizer = tf.train.AdamOptimizer(learningRate)
    gradients = optimizer.compute_gradients(loss)#cost
    train_op = optimizer.apply_gradients(gradients,global_step=global_step)
#取得batch
#自定义获取batch的方法
def get_batches(Xs,ys,batch_size):
    for start in range(0,len(Xs),batch_size):
        end = min(start+batch_size,len(Xs))
        yield Xs[start:end],ys[start:end]

