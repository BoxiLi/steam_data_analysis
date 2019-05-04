import numpy as np
import matplotlib.pyplot as plt
from data_preparation import *


def pow_normalize(mat):
    return mat/60**(1./3.)


class SVD_recommender(object):
    def __init__(self, mat, feature_num = 20, regu = 0.01):
        self.feature_num = feature_num
        # initialize U and I with an appropriate normalization
        self.UserFeature = np.matrix(np.random.rand(mat.shape[0], feature_num))*np.sqrt(2./feature_num)
        self.ItemFeature = np.matrix(np.random.rand(mat.shape[1], feature_num))*np.sqrt(2./feature_num)
        self.mat = mat
        self.useful_entry_number = mat.count_nonzero()
        self.regu = regu


    def reinitialize(self):
        self.UserFeature = np.matrix(np.random.rand(self.mat.shape[0], self.feature_num))*np.sqrt(2./self.feature_num)
        self.ItemFeature = np.matrix(np.random.rand(self.mat.shape[1], self.feature_num))*np.sqrt(2./self.feature_num)

    
    def svd_step(self, rate, row_indices, col_indices):
        rmse2 = 0.0
        for entry_iter in range(self.useful_entry_number):
            p = row_indices[entry_iter]
            q = col_indices[entry_iter]
            pred = np.dot(self.UserFeature[p], self.ItemFeature[q].T)[0,0]
            error = mat[p,q] - pred
            rmse2 = rmse2 + pow(error, 2)/self.useful_entry_number
             
            self.UserFeature[p] = self.UserFeature[p] + rate * (error * self.ItemFeature[q] - self.regu * self.UserFeature[p])
            self.ItemFeature[q] = self.ItemFeature[q] + rate * (error * self.UserFeature[p] - self.regu * self.ItemFeature[q])
        rmse = np.sqrt(rmse2)
        return rmse


    def optimize(self, step_num, rate, restart = False, parallel = True):
        """
        Run the optimization with given step number and learning rate. If previous optimization result exist, it 
        will continue with the existing user and game feature matrices.
        """
        if restart:
            self.reinitialize()
        last_rmse = np.inf
        learn_procress = [np.inf]
        row_indices, col_indices = mat.nonzero()
        for iter in range(step_num):
            rmse = self.svd_step(rate, row_indices, col_indices)
            learn_procress.append(rmse)
            print("rmse = ", rmse)
            last_rmse = rmse
            # if rmse start to fluctuate, we decrease the learning rate to get a better accuracy
            if rmse > last_rmse:
                rate = rate/2.
                print("The learning rate is decreased to", rate)
        return self.UserFeature, self.ItemFeature, learn_procress[1:]


file_name = "user_game30k"
generator = user_game_matrix(file_name)
generator.played_required = True
generator.thres_game = 20
generator.thres_user = 20
generator.normalize_func = tanh_normalize
mat, user_list, game_list = generator.construct()
recommender = SVD_recommender(mat, feature_num = 20, regu = 0.1)
UserFeature, ItemFeature, learn_procress = recommender.optimize(step_num = 500, rate = 0.01)


