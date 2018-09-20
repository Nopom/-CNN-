import tensorflow as tf
import pickle
# import cnn_movies.moviesRecommendafionGraph as cnnGraph
import cnn_movies.moviesRecommendationModel as cnnModel
import numpy as np
import os
#辅助函数
features = pickle.load(open('model/feature.p','rb'))
#features info: ['UserID' 'MovieID' 'Gender' 'Age' 'JobID' 'Title' 'Genres']
target_values = pickle.load(open('model/target.p','rb'))
title_count,title_set,genres2int,features,targets_values,ratings,users,\
movies,data,movies_orig,users_orig=pickle.load(open('model/params.p', 'rb'))
#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第五行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i,val in enumerate(movies.values)}
# print(movieid2idx)#{1: 0, 2: 1, 3: 2, 4: 3,.......
sentences_size = title_count #16
load_dir = './save'
movie_feature_size = user_feature_size = 200
movie_matrix_path = 'model/movie_matrix.p'
user_matrix_path = 'model/user_matrix.p'

#首先获取tensor
#使用函数 get_tensor_by_name()从 loaded_graph 中获取tensors
#'xxx:0'格式的xxx都是前面定义神经网络模型中的name
def get_tensors(load_graph):
    uid = load_graph.get_tensor_by_name('uid:0')
    user_gender = load_graph.get_tensor_by_name('user_gender:0')
    user_age = load_graph.get_tensor_by_name('user_age:0')
    user_job = load_graph.get_tensor_by_name('user_job:0')
    movie_id = load_graph.get_tensor_by_name('movie_id:0')
    movie_genres = load_graph.get_tensor_by_name('movie_genres:0')
    movie_titles = load_graph.get_tensor_by_name('movie_title:0')
    targets = load_graph.get_tensor_by_name('targets:0')
    dropout_keep_prob = load_graph.get_tensor_by_name('dropout_keep_prob:0')
    learning_rate = load_graph.get_tensor_by_name('learningRate:0')
    #维度(?,200)
    movie_combine_layer_flat = load_graph.get_tensor_by_name('movie_fc/Reshape:0')
    user_combine_layer_flat = load_graph.get_tensor_by_name('user_fc/Reshape:0')
    inference = load_graph.get_tensor_by_name('inference/ExpandDims:0')
    '''直接将用户的特征矩阵和电影特征矩阵相乘得到得分，最后要做的就是对着个得分进行回归
    # inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
    # inference = tf.expand_dims(inference, axis=1)'''
    return uid,user_gender,user_age,user_job,movie_id,movie_genres,\
            movie_titles,targets,dropout_keep_prob,learning_rate,inference,\
            movie_combine_layer_flat,user_combine_layer_flat

#指定用户和电影进行评分
#这部分就是网络的正向传播，计算得到预测的评分
def rating_movie(user_id,movie_id_val):
    load_graph = tf.Graph()
    with tf.Session(graph=load_graph) as sess:
        loder = tf.train.import_meta_graph(load_dir+'.meta')
        loder.restore(sess,load_dir)
        uid, user_gender, user_age, user_job, movie_id, movie_genres, \
        movie_titles, targets, dropout_keep_prob, learning_rate, inference, \
        movie_combine_layer_flat, user_combine_layer_flat = get_tensors(load_graph)
        genres = np.zeros([1,18])
        # print(movies)#MovieID,Title(15),genres（18）   MovieID从1开始
        genres[0] = movies.values[movieid2idx[movie_id_val]][2]
        titles = np.zeros([1,sentences_size])#15
        titles[0] = movies.values[movieid2idx[movie_id_val]][1]
        print('==',users.values)#uid,gender,age,job   uid从1开始，减一是为了符合下标索引
        feed = {
            uid:np.reshape(users.values[user_id-1][0],[1,1]),
            user_gender:np.reshape(users.values[user_id-1][1],[1,1]),
            user_age:np.reshape(users.values[user_id-1][2],[1,1]),
            user_job:np.reshape(users.values[user_id-1][3],[1,1]),
            #movieid2idx已经将下标匹配{1: 0, 2: 1, 3: 2, 4: 3,... 所以不需要-1
            movie_id:np.reshape(movies.values[movieid2idx[movie_id_val]][0],[1,1]),
            movie_genres:genres,
            movie_titles:titles,
            dropout_keep_prob:1
        }
        inference_val = sess.run([inference],feed)
        return inference_val

#生成movie特征矩阵，将训练好的电影数据特征组合成电影特征矩阵并保存到本地
#对每个电影进行正向传播
def save_movie_feature_matrix():
    load_graph = tf.Graph()
    movie_matrics = []
    with tf.Session(graph=load_graph) as sess:
        loader = tf.train.import_meta_graph(load_dir+'.meta')
        loader.restore(sess,load_dir)
        uid, user_gender, user_age, user_job, movie_id, movie_genres, \
        movie_titles, targets, dropout_keep_prob, learning_rate, inference, \
        movie_combine_layer_flat, user_combine_layer_flat = get_tensors(load_graph)
        for item in movies.values:
            genres = np.zeros([1,18])
            genres[0] = item.take(2)
            titles = np.zeros([1,sentences_size])
            titles[0] = item.take(1)
            feed = {
                movie_id:np.reshape(item.take(0),[1,1]),
                movie_genres:genres,
                movie_titles:titles,
                dropout_keep_prob:1
            }
            movie_representation = sess.run([movie_combine_layer_flat],feed)
            movie_matrics.append(movie_representation)
    movie_matrics = np.array(movie_matrics).reshape(-1,movie_feature_size)
    pickle.dump(movie_matrics,open(movie_matrix_path,'wb'))


#生成user特征矩阵
#将训练好的用户特征组合成用户特征矩阵并保存到本地
#对每个用户进行正向传播
def save_user_feature_matrix():
    load_graph = tf.Graph()
    users_matrics = []
    with tf.Session(graph=load_graph) as sess :
        loader = tf.train.import_meta_graph(load_dir+'.meta')
        loader.restore(sess,load_dir)
        uid, user_gender, user_age, user_job, movie_id, movie_genres, \
        movie_titles, targets, dropout_keep_prob, learning_rate, inference, \
        movie_combine_layer_flat, user_combine_layer_flat = get_tensors(load_graph)
        for item in users.values:
            feed = {
                uid:np.reshape(item.take(0),[1,1]),
                user_gender:np.reshape(item.take(1),[1,1]),
                user_age:np.reshape(item.take(2),[1,1]),
                user_job:np.reshape(item.take(3),[1,1]),
                dropout_keep_prob:1
            }
            user_representation = sess.run([user_combine_layer_flat],feed)
            users_matrics.append(user_representation)
    users_matrics = np.array(users_matrics).reshape(-1,user_feature_size)
    pickle.dump(users_matrics,open(user_matrix_path,'wb'))

def load_feature_matrix(path):
    if os.path.exists(path):
        pass
    elif path == movie_matrix_path:
        save_movie_feature_matrix()
    else:
        save_user_feature_matrix()
    return pickle.load(open(path,'rb'))

#使用电影特征矩阵推荐同类型的电影
#思路是计算指定电影的特征向量与整个电影特征矩阵的余弦相似度
#取相似度最大的top_k个
#ToDo：加入随机选择，保证每次的推荐稍微不同
def recommend_same_type_movie(movie_id,top_k = 5):
    load_graph = tf.Graph()
    movie_matrics = load_feature_matrix(movie_matrix_path)
    #给定电影的representation(代表)
    # print(movie_matrics.shape)
    movie_feature = movie_matrics[movieid2idx[movie_id]].reshape([1,movie_feature_size])
    # print(movie_feature.shape)
    with tf.Session(graph=load_graph) as sess:
        loader = tf.train.import_meta_graph(load_dir+'.meta')
        loader.restore(sess,load_dir)
        #计算余弦相似度
        print(movie_matrics.shape)#(3883,200)
        norm_movie_matrics = tf.sqrt(tf.reduce_sum(
            tf.square(movie_matrics),1,keep_dims=True))#计算每个representation的长度||x||
        print(norm_movie_matrics)#(3883,1)
        print(norm_movie_matrics[movie_id].shape)#(1,)
        print((norm_movie_matrics*norm_movie_matrics[movie_id]).shape)#(3883,1)
        normalized_movie_matrics = movie_matrics/(norm_movie_matrics*norm_movie_matrics[movie_id])
        print(normalized_movie_matrics.shape)#(3883,200)
        print(movie_feature.shape)#(1,200)
        probs_similarity = tf.matmul(movie_feature,tf.transpose(normalized_movie_matrics))
        # print(probs_similarity)#Tensor("MatMul:0", shape=(1, 3883), dtype=float32)
        #得到对于给定的movie_id，所有电影对它的余弦相似度
        #eval()将字符串str当成有效的表达式来求值并返回计算结果,
        # 通俗讲 就是通过str的格式将其定义为这种格式的数据类型（list，dict...）
        sim = probs_similarity.eval()
        print(sim)#numpy.ndarray
    print('和电影：{}相似的电影有：\n'.format(movies_orig[movieid2idx[movie_id]][1]))
    sim = np.squeeze(sim) #将二维sim转为一维
    print(sim)
    #argsort函数返回的是数组值从小到大的索引值,加负号表示降序
    res_list = np.argsort(-sim)[:top_k]#获取余弦相似度最大的前top_k个movie信息
    results = list()
    for res in res_list:
        movie_info = movies_orig[res]
        results.append(movie_info)
    return results

#看过这个电影的人还可能（喜欢）那些电影
#首先选出喜欢摸个电影的top_k个人，得到这几个人的用户特征向量
#然后计算这几个人对所有电影的评分
#选择每个人评分最高的电影最为推荐
#TODO 加入随机选择
def recommend_your_favorite_movie(user_id,top_k=5):
    load_graph = tf.Graph()
    movie_matircs = load_feature_matrix(movie_matrix_path)
    users_matircs = load_feature_matrix(user_matrix_path)
    #是否需要减一
    user_feature = users_matircs[user_id-1].reshape([1,user_feature_size])
    with tf.Session(graph=load_graph) as sess:
        loader = tf.train.import_meta_graph(load_dir+'.meta')
        loader.restore(sess,load_dir)
        #获取图中的inference，然后用sess运行
        probs_similarity = tf.matmul(user_feature,tf.transpose(movie_matircs))
        sim = probs_similarity.eval()
        sim = np.squeeze(sim)
        print(sim)
        #获取该用户对所有电影可能评分最高的top_k
        # argsort函数返回的是数组值从小到大的索引值
        res_list = np.argsort(-sim)[:top_k]
        results = []
        for res in res_list:
            movie_info = movies_orig[res]
            results.append(movie_info)
        # print('以下是给您的推荐：',results)
        return results

def recommend_other_favorite_movie(movie_id,top_k=5):
    load_graph = tf.Graph()
    movie_matrics = load_feature_matrix(movie_matrix_path)
    users_matrics = load_feature_matrix(user_matrix_path)
    movie_feature = (movie_matrics[movieid2idx[movie_id]]).reshape([1,movie_feature_size])
    print('您看的电影是：{}'.format(movies_orig[movieid2idx[movie_id]]))
    with tf.Session(graph=load_graph) as sess:
        loader = tf.train.import_meta_graph(load_dir+'.meta')
        loader.restore(sess,load_dir)
        #计算对给定movie，所有用户对其可能的评分
        users_inference = tf.matmul(movie_feature,tf.transpose(users_matrics))
        sim = users_inference.eval()
        sim = np.squeeze(-sim)
        print(sim)
        favorite_users_id = np.argsort(-sim)[:top_k]
        print(favorite_users_id)
        #user_id处理时是否需要减一
        print('喜欢看这个电影的人是：{}'.format(users_orig[favorite_users_id-1]))
        result = []
        for user in favorite_users_id:
            movies = recommend_your_favorite_movie(user,top_k=5)
            result.extend(movies)
        # print('喜欢这个电影的人还喜欢：',result)
        return  result

#在这里测试每个推荐功能
#预测给定user对给定movie的评分
# inference_val = rating_movie(123,1234)
# print('for user:123, predicting the rating for movie:1234', inference_val[0][0][0])

#生成user和movie的特征矩阵，并存储到本地
# save_movie_feature_matrix()
# save_user_feature_matrix()
#对给定的电影，推荐相同类型的其他top_k个电影
results = recommend_same_type_movie(movie_id=222,top_k=5)
for i in results:
    print(i)
# # 对给定用户，推荐相同可能喜欢的top_k个电影
# results = recommend_your_favorite_movie(user_id=222,top_k=5)
# for i in results:
#     print(i)
# # # 看过这个电影的人还可能喜欢看哪些电影
# results = recommend_other_favorite_movie(movie_id=222,top_k=5)
# for i in results:
#     print("好",i)