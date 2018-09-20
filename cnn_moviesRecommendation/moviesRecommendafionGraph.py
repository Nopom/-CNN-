import tensorflow as tf
import matplotlib.pyplot as plt
import time
import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cnn_movies.moviesRecommendationModel as cnnModel


# Number of Epochs
num_epochs = 2
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 50



#训练网络
train_graph = cnnModel.train_graph
gradients = cnnModel.gradients
loss = cnnModel.loss
losses = {'train':[],"test":[]}
with tf.Session(graph=train_graph) as sess:
    #搜索数据给tensorBoard用
    grad_summaries = []
    for g,v in gradients:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':',"_")),g)
            # tf.nn.zero_fraction 用于计算矩阵中 0 所占的比重，也就是计算矩阵的稀疏程度
            sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name.replace(':','_')),\
                                                 tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir,'runs',timestamp)))
    print('Writing to {}\n'.format(out_dir))

    loss_summary = tf.summary.scalar('loss',loss)
    #train summaries
    train_summary_op = tf.summary.merge([loss_summary,grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir,'summaries','train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir,sess.graph)
    #inference summaries
    inference_summary_op = tf.summary.merge([loss_summary])
    inference_summary_dir = os.path.join(out_dir,'summaries','inference')
    inference_summary_writer = tf.summary.FileWriter(inference_summary_dir,sess.graph)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch_i in range(num_epochs):
        print('第一次训练：',epoch_i)
        #将数据集分成训练集和测试集，随机种子不固定
        train_X, test_X, train_y, test_y = train_test_split(cnnModel.features,\
                                cnnModel.targets_values,test_size=0.2,random_state=0)
        train_batches = cnnModel.get_batches(train_X,train_y,batch_size)
        test_batches = cnnModel.get_batches(test_X,test_y,batch_size)
        #训练的迭代，保存训练损失
        for batch_i in range(len(train_X)//batch_size):
            # features特征 UserID', 'MovieID','Gender','Age','JobID','Title', 'Genres'
            x,y = next(train_batches)
            genres = np.zeros([batch_size,18])
            for i in range(batch_size):
                genres[i] = x.take(6,1)[i]
            titles = np.zeros([batch_size,cnnModel.sentence_size])
            for i in range(batch_size):
                titles[i] = x.take(5,1)[i]
            feed = {
                cnnModel.uid:np.reshape(x.take(0,1),[batch_size,1]),
                cnnModel.user_gender:np.reshape(x.take(2,1),[batch_size,1]),
                cnnModel.user_age:np.reshape(x.take(3,1),[batch_size,1]),
                cnnModel.user_job:np.reshape(x.take(4,1),[batch_size,1]),
                cnnModel.movie_id:np.reshape(x.take(1,1),[batch_size,1]),
                cnnModel.movie_genres:genres,#x.take(6,1)
                cnnModel.movie_title:titles,#x.take(5,1)
                cnnModel.targets:np.reshape(y,[batch_size,1]),
                cnnModel.dropout_keep_prob:dropout_keep,
                cnnModel.learningRate:learning_rate}
            step,train_loss,summaries,_ = sess.run([cnnModel.global_step,loss,\
                                        train_summary_op,cnnModel.train_op],feed)
            # train_loss,_ = sess.run([loss,\
            #                             cnnModel.train_op],feed)
            #保存测试损失
            losses['train'].append(train_loss)
            train_summary_writer.add_summary(summaries,step)

            if batch_i % show_every_n_batches == 0:
                time_str = datetime.datetime.now().isoformat()
                print("{}:Epoch {:>3} Batch {:>4}/{} train_loss = {:.3f}".format(
                    time_str,epoch_i,batch_i,(len(train_X)//batch_size),train_loss))
        # 使用测试数据的迭代
        for batch_i in range((len(test_X)//batch_size)):
            x,y = next(test_batches)
            genres = np.zeros([batch_size,18])
            for i in range(batch_size):
                genres[i] = x.take(6,1)[i]#(256,18)
            titles = np.zeros([batch_size,cnnModel.sentence_size])
            for i in range(batch_size):
                titles[i] = x.take(5,1)[i]
            feed = {
                cnnModel.uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                cnnModel.user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                cnnModel.user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                cnnModel.user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                cnnModel.movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                cnnModel.movie_genres: genres,  # x.take(6,1)
                cnnModel.movie_title: titles,  # x.take(5,1)
                cnnModel.targets: np.reshape(y, [batch_size, 1]),
                cnnModel.dropout_keep_prob: 1,
                cnnModel.learningRate: learning_rate}
            step,test_loss,summaries = sess.run([cnnModel.global_step,loss,inference_summary_op],feed)
            #保存测试损失
            losses['test'].append(test_loss)
            inference_summary_writer.add_summary(summaries,step)
            time_str = datetime.datetime.now().isoformat()
            if batch_i % show_every_n_batches == 0:
                print("{}: Epoch {:>3} Batch {:>4}/{} test_loss = {:.3f}".format(
                    time_str,epoch_i,batch_i,(len(test_X)//batch_size),test_loss))
    #保存模型
    saver.save(sess,cnnModel.save_dir)
    print('Model Trained and Saved')
