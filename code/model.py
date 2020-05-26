import tensorflow as tf
from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from utils import *


class GCNModel():

    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, num_r, name, act=tf.nn.elu):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        self.att = tf.Variable(tf.constant([0.5, 0.33, 0.25]))
        self.num_r = num_r
        with tf.variable_scope(self.name):
            self.build()

    def build(self):
        self.adj = dropout_sparse(self.adj, 1-self.adjdp, self.adj_nonzero)

        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden2 = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden1)

        self.emb = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden2)

        self.embeddings = self.hidden1 * \
            self.att[0]+self.hidden2*self.att[1]+self.emb*self.att[2]

        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, act=tf.nn.sigmoid)(self.embeddings)
