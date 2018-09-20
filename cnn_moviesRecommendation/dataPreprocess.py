import pandas as pd
import numpy as np
import re
import pickle
#==========================1.数据加载=====================================
def data_down(data_dir):
    # 设置列名
    users_title = ['UserID','Gender','Age','JobID','Zip-code']#ID,性别，年龄，职业ID，邮编
    movies_title = ['MovieID', 'Title', 'Genres']#ID,电影名，流派
    ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
    #读入数据
    users = pd.read_table(data_dir+'users.dat',sep='::',engine='python',names=users_title)
    movies = pd.read_table(data_dir+'movies.dat',sep='::',engine='python',names=movies_title)
    radings = pd.read_table(data_dir+'ratings.dat',sep='::',engine='python',names=ratings_title)
    return users,movies,radings
users,movies,ratings = data_down('ml-1m/')


#=====================数据预处理========================================

# UserID、Occupation和MovieID不用变。
# Gender字段：需要将‘F’和‘M’转换成0和1。
# Age字段：要转成7个连续数字0~6。

#流派 Genres字段：是分类字段，要转成数字。首先将Genres中的类别转成字符串到数字的字典，
# 然后再将每个电影的Genres字段转成数字列表，因为有些电影是多个Genres的组合。

# Title字段：处理方式跟Genres字段一样，首先创建文本到数字的字典，
# 然后将Title中的描述转成数字的列表。另外Title中的年份也需要去掉。

# Genres和Title字段需要将长度统一，这样在神经网络中方便处理。
# 空白部分用‘< PAD >’对应的数字填充。

def data_preprocessing(users,movies,ratings):

    print(users.shape)
    print(movies.shape)
    print(ratings.shape)

    # users处理
    # 去掉zip-code字段
    users = users.filter(regex='UserID|Gender|Age|JobID')
    #没有经过处理的users数据
    users_orig = users.values
    #将性别Gender转为0,1格式
    gender_map = {'F':0,'M':1}
    users['Gender'] = users['Gender'].map(gender_map)
    #将年龄编号，将字段值改为对应编号
    age_map = {val:i for i,val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    #movies处理
    #没有处理过的movies数据
    movies_orig = movies.values
    #去掉title中的年份
    pattern = re.compile(r'^(.*)\((\d+)\)')
    title_map = {val:pattern.match(val).group(1) for i,val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)
    #将电影类型转为数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)
    #将类型进行编号
    genres_set.add('<PAD>')
    genres2int = {val:i for i,val in enumerate(genres_set)}#内容'Sci-Fi': 0, 'Animation': 1
    #将电影类型转为等长数字列表 内容'Action|Crime|Mystery': [4, 13, 3]
    gender_map = {val:[genres2int[row] for row in val.split('|')] for i,val in enumerate(set(movies['Genres']))}
    #将每个样本的电影类型处理成相同长度，不够用<PAD>填充
    for key in gender_map:
        for cnt in range(max(genres2int.values())-len(gender_map[key])):
            gender_map[key].insert(len(gender_map[key])+cnt,genres2int['<PAD>'])
    movies['Genres'] = movies['Genres'].map(gender_map)
    #电影Title转数字字典
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    title_set.add('<PAD>')
    title2int = {val:i for i,val in enumerate(title_set)}
    #将电影Title转成等长的数字列表，长度15
    title_count = 15
    title_map = {val:[title2int[row] for row in val.split()] for i,val in enumerate(set(movies['Title']))}
    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key])+cnt,title2int['<PAD>'])
    movies['Title'] = movies['Title'].map(title_map)

    #ratings处理
    #去掉时间字段timestamps
    ratings = ratings.filter(regex='UserID|MovieID|Rating')


    #合并三个表
    data = pd.merge(pd.merge(ratings,users),movies)
    #将数据分成X和y两张表,特征和目标标签（评分）
    target_fields = ['Rating']
    features_pd,target_pd = data.drop(target_fields,axis=1),data[target_fields]
    features = features_pd.values
    target_values = target_pd.values


    # title_count：Title字段的长度（15）
    # title_set：Title文本的集合
    # genres2int：电影类型转数字的字典
    # features：是输入X
    # targets_values：是学习目标y
    # ratings：评分数据集的Pandas对象
    # users：用户数据集的Pandas对象
    # movies：电影数据的Pandas对象
    # data：三个数据集组合在一起的Pandas对象
    # movies_orig：没有做数据处理的原始电影数据
    # users_orig：没有做数据处理的原始用户数据
    return title_count,title_set,genres2int,features,target_values,\
           ratings,users,movies,data,movies_orig,users_orig

title_count, title_set, genres2int, features, targets_values, \
        ratings, users, movies, data, movies_orig, users_orig = data_preprocessing(users,movies,ratings)
# 用pickle保存数据到文件
# with open('cnn_movies/ml-1m/preprocess.pkl','wb') as f:
#     pickle.dump((title_count,title_set,genres2int,
#                  features,targets_values,ratings,
#                  users,movies,data,
#                  movies_orig,users_orig),f)
f = open('model/feature.p','wb')
pickle.dump(features,f)
f = open('model/target.p','wb')
pickle.dump(targets_values,f)
f = open('model/params.p','wb')
pickle.dump((title_count,title_set,genres2int,features,targets_values,\
            ratings,users,movies,data,movies_orig,users_orig),f)
title_vocb_num = len(title_set)+1
genres_num = len(genres2int)
movie_id_num = max(movies['MovieID'])+1
f = open('model/argument.p','wb')
pickle.dump((movie_id_num,title_count,title_vocb_num,genres_num),f)
print(title_vocb_num,genres_num,movie_id_num)
print("数据保存成功！！！")