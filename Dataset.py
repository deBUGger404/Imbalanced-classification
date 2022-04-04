

import tensorflow as tf
import random as rn
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder
import joblib

sd = 123
np.random.seed(sd)
rn.seed(sd)
os.environ['PYTHONHASHSEED']=str(sd)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#read files image path files
train = pd.read_csv("GroceryStoreDataset/dataset/train.txt", sep=",", header=None, names= ['img_path','fine_label','coarse_label'])  
test = pd.read_csv("GroceryStoreDataset/dataset/test.txt", sep=",", header=None, names= ['img_path','fine_label','coarse_label'])  
val = pd.read_csv("GroceryStoreDataset/dataset/val.txt", sep=",", header=None, names= ['img_path','fine_label','coarse_label'])  
classes = pd.read_csv('GroceryStoreDataset/dataset/classes.csv')

# create variable for packed type: packed is 1 and loose is 0
train['product'] = train['img_path'].apply(lambda x: 1 if 'Packages' in x else 0)
test['product'] = test['img_path'].apply(lambda x: 1 if 'Packages' in x else 0)
val['product'] = val['img_path'].apply(lambda x: 1 if 'Packages' in x else 0)

def one_hot(df, ohe):
    feature = ohe.transform(df[['product','coarse_label']])
    features = pd.DataFrame(feature, columns=list(ohe.get_feature_names()))
    df1 = pd.concat([df,features],axis=1)
    return df1

try:
    # save one hot encoded file if not than the below code will run
    ohe = joblib.load('ohe_final.pkl')
except:
    ohe = OneHotEncoder(sparse=False)
    feature_arr = ohe.fit_transform(train[['product','coarse_label']])
    joblib.dump(ohe,'ohe_final.pkl')
train1 = one_hot(train, ohe)
val1 = one_hot(val, ohe)
test1 = one_hot(test, ohe)

cols = ohe.get_feature_names()

# image phrasing function for data cleaning and normaization
def img_prase(image_paths, label, len_):
    image_content = tf.io.read_file('GroceryStoreDataset/dataset/' + image_paths)
    image = tf.image.decode_jpeg(image_content, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    images = tf.image.resize(image,[224,224], method='bicubic')
    return images, label

# data batch creation and transformation
def data_batch(df, cols, product = 'all', shuffle=None, batch_size=1):
    # df is input data like train1, test1, val1
    # cols: which columns we have to use select for label
    # product type: all for whole data packed+loose data, packed for data which contanis packed data and loose for data which contains loose data
    # shuffle: to shuffle data like train data for training
    # batch size: how many batches you want to pass in model
    if product == 'packages': df1 = df[df['product']==1]
    elif product == 'loose': df1 = df[df['product']==0]
    elif product == 'all': df1 = df
    else: 
        print('please specify product type')
    image_list, label_list, coarse_id = df1['img_path'].values, df1[cols[:2]].values, df1[cols[2:]].values
    num_sample = len(image_list)
    len_ = len(cols[2:])
    print(num_sample, np.unique(label_list[:,1],return_counts=True),np.unique(coarse_id).shape[0],len_)
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    if product=='all':
        labels = tf.convert_to_tensor(label_list, dtype=tf.float32)
    else:
        labels = tf.convert_to_tensor(coarse_id, dtype=tf.float32)

    data = tf.data.Dataset.from_tensor_slices((images, labels))

    data = data.map(lambda x, y: img_prase(x,y,len_), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        data = data.shuffle(num_sample).batch(batch_size).repeat()
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        data = data.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data
