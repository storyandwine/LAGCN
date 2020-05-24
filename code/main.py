import numpy as np
import scipy.sparse as sp
import scipy.io as spio
import tensorflow as tf
import gc
import random
import math
import os
import requests
from clr import cyclic_learning_rate
from clac_metric import cv_model_evaluate

def constructAdjNet(drug_dis_matrix):
    drug_matrix = np.matrix(np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]),dtype=np.int8))

    mat1 = np.hstack((drug_matrix,drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T,dis_matrix))
    adj =  np.vstack((mat1,mat2))
    # adj =  adj + sp.eye(adj.shape[0])
    return adj

def constructXNet(drug_dis_matrix,drug_matrix,dis_matrix):
    mat1 = np.hstack((drug_matrix,drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T,dis_matrix))
    return np.vstack((mat1,mat2))

def Get_embedding_Matrix(train_drug_dis_matrix,drug_matrix,dis_matrix,seed,epochs,dp,w,lr,drug_dis_matrix,adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    # adj=constructAdjNet(train_drug_dis_matrix)
    adj=constructXNet(train_drug_dis_matrix,drug_matrix,dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_dis_matrix.sum()
    # num_nodes = adj.shape[0]
    # num_edges = adj.sum()
    # F1 = drug_matrix
    # F2 = dis_matrix
    # sim_mat = np.vstack((np.hstack((F1,np.zeros(shape=(F1.shape[0],F2.shape[1]),dtype=int))), np.hstack((np.zeros(shape=(F2.shape[0],F1.shape[1]),dtype=int), F2))))
    # sim_mat = sp.coo_matrix(sim_mat)
    # F1 = drug_matrix
    # F2 = dis_matrix
    # X = np.vstack((np.hstack((F1,np.zeros(shape=(F1.shape[0],F2.shape[1]),dtype=int))), np.hstack((np.zeros(shape=(F2.shape[0],F1.shape[1]),dtype=int), F2))))
    # X = constructXNet(train_drug_dis_matrix,drug_matrix,dis_matrix)
    X = constructAdjNet(train_drug_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    # features = sparse_to_tuple(sp.identity(num_nodes))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))     
    # adj_orig.eliminate_zeros()
    
    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, features_nonzero,adj_nonzero, name='yeast_gcn')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            w=w,lr=lr,association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _,avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        
        if epoch%100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            metric_tmp = roc_auc_score(drug_dis_matrix.flatten(),res)
            # metric_tmp = cv_model_evaluate(drug_dis_matrix,res.reshape((269,598)), train_drug_dis_matrix)
            print("Epoch:", '%04d' % (epoch + 1), 
            "train_loss=", "{:.5f}".format(avg_cost),
            "score=")
            print(metric_tmp)
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    att = sess.run(model.att,feed_dict=feed_dict)
    with open('resultDNN/att.csv','ab') as f: 
        np.savetxt(f, att.reshape(1,3), delimiter=",")
    sess.close()
    return res

def matrix_normalize(similarity_matrix):
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        similarity_matrix = similarity_matrix - np.diag(np.diag(similarity_matrix))
        for i in range(200):
            D = np.diag(np.array(np.sum(similarity_matrix, axis=1)).flatten())
            D = np.linalg.pinv(np.sqrt(D))
            similarity_matrix = D * similarity_matrix * D
    else:
        for i in range(similarity_matrix.shape[0]):
            if np.sum(similarity_matrix[i], axis=1) == 0:
                similarity_matrix[i] = similarity_matrix[i]
            else:
                similarity_matrix[i] = similarity_matrix[i] / np.sum(similarity_matrix[i], axis=1)
    return similarity_matrix

def cross_validation_experiment(drug_dis_matrix,drug_matrix,dis_matrix,seed,epochs,dp,w,lr,adjdp):
    index_matrix = np.mat(np.where(drug_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam % k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating drug-disease...." % (seed))
    # flag = 0
    for k in range(k_folds):
        print("------this is %dth cross validation------"%(k+1))
        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        drug_len = drug_dis_matrix.shape[0]
        dis_len = drug_dis_matrix.shape[1]
        # drug_disease_res = SCMF(train_matrix, drug_matrix, np.mat(dis_matrix), 1, 4, int(269 * 0.45))
        drug_disease_res = Get_embedding_Matrix(train_matrix,drug_matrix,dis_matrix,seed,epochs,dp,w,lr,drug_dis_matrix,adjdp)
        
        predict_y_proba = drug_disease_res.reshape(drug_len,dis_len)
        # predict_y_proba = sigmoid_array(predict_y_proba)
        # predict_y_proba = np.array(drug_disease_res)
        metric_tmp = cv_model_evaluate(drug_dis_matrix,predict_y_proba, train_matrix)
        print(metric_tmp)
        metric += metric_tmp
        # if metric_tmp[0]<0.3 or metric_tmp[1]<0.8:
        #     flag = k+1
        #     break
        
        
        del train_matrix
        
        gc.collect()
    
    # print("-------AVG-------")
    print(metric / k_folds)
    # if flag!=0:
    #     metric = np.array(metric / flag)
    # else:
    metric = np.array(metric / k_folds)
    # with open('resultDNN/ParameterSetting.csv','a') as f: 
        # f.write('act=elu 64D 3layer att weightedlossF = adj clr'+str(lr)+' epoch='+str(epoch)+'posw='+str(w)+'adjdp='+str(adjdp)+'dp='+str(dp)+'simw='+str(simw)+'seed='+str(seed)+',')
        # 'simw='+str(simw)+  '+str(drug)+'
    with open('resultDNN/ParameterSetting.csv','ab') as f: 
        np.savetxt(f, metric, delimiter=",")
    return metric

def get_Jaccard2_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    E = np.ones_like(X.T)
    denominator=X * E + E.T * X.T - X * X.T
    denominator_zero_index=np.where(denominator==0)
    denominator[denominator_zero_index]=1
    result = X * X.T / denominator
    result[denominator_zero_index]=0
    result = result - np.diag(np.diag(result))
    return matrix_normalize(result)


if __name__=="__main__":
    similarity_data = spio.loadmat('similarity.mat')
    scmfdd = spio.loadmat('SCMFDD_Dataset.mat')
    drug_dis_matrix = scmfdd['drug_disease_association_matrix']
    # drug_features = ['structure_feature_matrix', 'target_feature_matrix','enzyme_feature_matrix','pathway_feature_matrix', 'drug_drug_interaction_feature_matrix']
    drug_features = ['target_feature_matrix']
    epoch =4000
    lr = 0.01
    adjdp = 0.6
    dp =  0.4
    simw = 6
    drug_sim =
    dis_sim = np.array(similarity_data['normalized_dis_similairty_matrix'])
    
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    for j in range(circle_time):
        result += cross_validation_experiment(drug_dis_matrix,drug_sim*simw,dis_sim*simw,j,epoch,dp,lr,adjdp)
    average_result = result / circle_time
    print(average_result)