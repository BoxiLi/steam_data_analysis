import numpy as np
import matplotlib.pyplot as plt
from data_preparation import *


def pow_normalize(mat):
    return mat/60**(1./3.)


class SVD_recommender(object):
    def __init__(self, mat, feature_num = 20, regu = 0.01):
        self.feature_num = feature_num
        # initialize U and I with an appropriate normalization
        self.UserFeature = np.random.rand(mat.shape[0], feature_num)*np.sqrt(2./feature_num)/10.
        self.ItemFeature = np.random.rand(mat.shape[1], feature_num)*np.sqrt(2./feature_num)/10.
        self.mu = np.mean(mat)
        self.UserUncertain = np.random.rand(mat.shape[0])*np.sqrt(2./feature_num)/10.
        self.ItemUncertain = np.random.rand(mat.shape[1])*np.sqrt(2./feature_num)/10.
        self.mat = mat
        self.useful_entry_number = mat.count_nonzero()
        self.regu = regu


    def reinitialize(self):
        self.UserFeature = np.matrix(np.random.rand(self.mat.shape[0], self.feature_num))*np.sqrt(2./self.feature_num)
        self.ItemFeature = np.matrix(np.random.rand(self.mat.shape[1], self.feature_num))*np.sqrt(2./self.feature_num)

    
    def svd_step(self, rate, row_indices, col_indices):
        """
        one svd step, it goes through all the non-zero entries of mat
        """
        rmse2 = 0.0
        for entry_iter in range(self.useful_entry_number):
            p = row_indices[entry_iter]
            q = col_indices[entry_iter]
            pred = self.mu + self.UserUncertain[p] + self.ItemUncertain[q] + np.dot(self.UserFeature[p].T, self.ItemFeature[q])
            error = self.mat[p,q] - pred
            rmse2 = rmse2 + pow(error, 2)/self.useful_entry_number
            
            self.UserUncertain[p] = self.UserUncertain[p] + rate * (error - self.regu * self.UserUncertain[p])
            self.ItemUncertain[q] = self. ItemUncertain[q] + rate * (error - self.regu * self.ItemUncertain[q])
            self.UserFeature[p] = self.UserFeature[p] + rate * (error * self.ItemFeature[q] - self.regu * self.UserFeature[p])
            self.ItemFeature[q] = self.ItemFeature[q] + rate * (error * self.UserFeature[p] - self.regu * self.ItemFeature[q])
        rmse = np.sqrt(rmse2)
        return rmse


    def optimize(self, step_num, rate, restart = False):
        """
        Optimize the user feature and item feature matrices. If previous result exists, it will continue with it.
        """
        if restart:
            self.reinitialize()
        learn_procress = [np.inf]*5 # inf just for compare, will be deleted at the end
        row_indices, col_indices = self.mat.nonzero()
        for i in range(step_num):
            rmse = self.svd_step(rate, row_indices, col_indices)
            learn_procress.append(rmse)
            print("rmse = ", rmse)
            if rmse > learn_procress[-5]:
                break
        return self.UserFeature, self.ItemFeature, learn_procress[5:]

    
    def dig_hole(self, num_samples):
        # dig holes, pick random coordinates in mat and set them to 0
        row_indices, col_indices = self.mat.nonzero()
        num_total_entries = len(row_indices)

        pointers = np.random.choice(range(num_total_entries), num_samples, replace=False) # replace = False, no repetition
        hole_indices = list(zip(row_indices[pointers], col_indices[pointers])) # create coordiante pairs [(r1,c1),(r2,c2)...]
        hole_values = [mat[index] for index in hole_indices] # record existing mat entries
        for index in hole_indices:
            mat[index] = 0. # set to 0, delete it from the sparse mat so that it will be fitted)
        return {hole_indices[i]:hole_values[i] for i in range(num_samples)}


if __name__ == "__main__":
    file_name = "user_game30k"
    generator = user_game_matrix(file_name)
    generator.played_required = True
    generator.thres_game = 20
    generator.thres_user = 20
    generator.normalize_func = tanh_normalize
    mat, user_list, game_list = generator.construct()
    recommender = SVD_recommender(mat, feature_num = 20, regu = 0.01) 
    UserFeature, ItemFeature, learn_procress = recommender.optimize(step_num = 13, rate = 0.01)