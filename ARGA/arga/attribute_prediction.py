from __future__ import division
from __future__ import print_function
import os
import numpy as np
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import settings
from constructor import get_placeholder, get_model, format_data, get_optimizer, update
from constructor_attr import format_data_attr
from metrics import linkpred_metrics

# For attrpred_metrics_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from metrics import attrpred_metrics_classi

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class Attr_pred_Runner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']
        self.n_hop_enable = settings['n_hop_enable']                      # Author: Tonni
        self.hop_count = settings['hop_count']                            # Author: Tonni
        self.pred_column = settings['pred_column']                        # Author: Tonni


    def erun(self):
        print('Attribute prediction ...')

        model_str = self.model

        # formatted data
        feas = format_data(self.data_name, self.n_hop_enable, self.hop_count, self.pred_column)             # Updated with self.n_hop_enable and hop_count and pred_column
        # Only for attribute prediction, we need this 
        y_tobe_predicted = feas['y_tobe_predicted']                       # Author: Tonni                   # array

        # Define placeholders
        placeholders = get_placeholder(feas['adj'])

        # construct model
        d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])

        # Initialize session
        sess = tf.Session()
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(tf.global_variables_initializer())

        # -------------------------------------------------------------------------------------------------------------
        # # get attribute 0 as y                               # Author: Tonni - Right now I dont need this portion
        # target_attribute = 0
        # y = np.zeros(feas['num_nodes'])

        # for i in range(feas['features_nonzero']):
        #     row = feas['features'][0][i][0]
        #     col = feas['features'][0][i][1]
        #     if(col == target_attribute):
        #         y[row] = 1 # feas['features'][1][i] # always 1

        # print()                                               # Author: Tonni
        # print(y.shape, y.size)                                # Author: Tonni
        # print()                                               # Author: Tonni
        # -------------------------------------------------------------------------------------------------------------
        y = y_tobe_predicted                                  # Author: Tonni
        # Train model

        val_roc_score = []

        prev_loss = 1.0
        prev_emb = None

        for epoch in range(self.iteration):

            reconstruct_loss = 0
            
            for i in range(50):
                emb, reconstruct_loss = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])
                # stop when loss no longer decreases
                if prev_loss < reconstruct_loss:
                    print('force stopping, epoch:', epoch, 'reconstruct_loss:', reconstruct_loss)
                    emb = prev_emb
                    reconstruct_loss = prev_loss
                    break

                prev_emb = emb
                prev_loss = reconstruct_loss

            # print("Using final emb for attribute prediction...", "emb.shape: ", emb.shape)
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(reconstruct_loss))

            # Run classification model on the dataset if predicting attribute is binary
            y_test, y_pred = self.classify(emb, y)

            if (epoch+1) % 10 == 0:
                roc_score, ap_score = attrpred_metrics_classi(y_test, y_pred)
                print('Test ROC score: ' + str(roc_score))
                print('Test AP score: ' + str(ap_score))
            
    def classify(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
        reg_log = LogisticRegression()
        reg_log.fit(X_train, y_train)
        y_pred = reg_log.predict(X_test)
        return y_test, y_pred

    def multilabel_classify(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
        reg_log = LogisticRegression()
        reg_log.fit(X_train, y_train)
        y_pred = reg_log.predict(X_test)
        return y_test, y_pred

    def regression(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
        reg_log = LogisticRegression()
        reg_log.fit(X_train, y_train)
        y_pred = reg_log.predict(X_test)
        return y_test, y_pred









            # # print after every 10 epochs
            # if epoch % 10 == 0:
            #     print('epoch:', epoch, 'reconstruct_loss:', reconstruct_loss)

            # # stop when loss no longer decreases
            # if prev_loss < reconstruct_loss:
            #     print('force stopping, epoch:', epoch, 'reconstruct_loss:', reconstruct_loss)
            #     emb = prev_emb
            #     reconstruct_loss = prev_loss
            #     break
            
            # if (epoch+1) % 2 == 0:
            #     kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(emb)
            #     print("Epoch:", '%04d' % (epoch + 1))
            #     predict_labels = kmeans.predict(emb)
            #     cm = clustering_metrics(feas['true_labels'], predict_labels)
            #     cm.evaluationClusterModelFromLabel()        

        # print("Using final emb for attribute prediction...")
        # print(emb.shape)
        # print(emb)
        # Run classification model on the dataset if predicting attribute is binary
        # self.classify(emb, y)

        # Run classification model on the dataset if predicting attribute is non-binary

    
    # def classify(self, X, y):

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    #     reg_log = LogisticRegression()
    #     reg_log.fit(X_train, y_train)
    #     y_pred = reg_log.predict(X_test)
    #     return y_test, y_pred
        # print(metrics.classification_report(y_test, y_pred))
        # print(y_test.shape)
        # print("------------")
        # print(y_pred.shape)