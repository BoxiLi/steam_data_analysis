import numpy as np
import matplotlib.pyplot as plt
from data_preparation import *
import csv



def pow_normalize(mat):
    return mat/60**(1./3.)


def SVD(mat, feature = 20, step = 300, Rate = 0.0001, Type = 0, ItemFeature = [0]):
    
    ARmse = np.inf
    Rmse = 0.0
    lr = Rate
    Lambda = 0.01
    L = [[],[]]
    error = 0.0
    x=0.0
    np.random.seed(0)
    UserFeature = np.matrix(np.random.rand(mat.shape[0], feature)*np.sqrt(2/feature))
    if Type == 0:
        ItemFeature = np.matrix(np.random.rand(mat.shape[1], feature)*np.sqrt(2/feature))
    row_indices, col_indices = mat.nonzero() # Returns a tuple of arrays (row,col) containing the indices of the non-zero elements of the matrix.
    useful_entry_number = len(row_indices)
    for i in range(step):
        rmse = 0.0
        for entry_iter in range(useful_entry_number):
            p = row_indices[entry_iter]
            q = col_indices[entry_iter]
            Pred = np.dot(UserFeature[p,:], ItemFeature[q,:].T)
            error = mat[p,q] - Pred
            #print("error = ",error)
            rmse = rmse + pow(error/np.sqrt(useful_entry_number), 2)
            #type0 for the whole data set, type1 for new userdata.  
            if Type == 0:
                UserFeature[p] = UserFeature[p] + lr * (error * ItemFeature[q] - Lambda * UserFeature[p])
                ItemFeature[q] = ItemFeature[q] + lr * (error * UserFeature[p] - Lambda * ItemFeature[q])
            elif (Type == 1):
                UserFeature[p] = UserFeature[p] + lr * (error * ItemFeature[q])
       #type0 for the whole data set, type1 for new userdata.                     
        Rmse = np.sqrt(rmse)
        print("Rmse = ", Rmse, "ARmse = ", ARmse)
        L[0].append(x+i*lr)
        L[1].append(Rmse)
        if Rmse < ARmse:
            ARmse = Rmse
       #     lr = lr*0.99
       #     lr = lr/(1.0 + 0.0001*i)
       # if abs(Rmse)<0.1:
        else:
            break
    LearningProcess = L
    #Result = np.dot(UserFeature, ItemFeature.T)
    return UserFeature, ItemFeature, LearningProcess,  ARmse

def main(Feature = 19, Step = 300, rate = 0.001):
    file_name = "user_game30k"
    file_name2 = "user_game300k"  # providing data for evaluating
    generator = user_game_matrix(file_name)
# default of "played_required" is True
    generator.played_required = True
    generator.thres_game = 20
    generator.thres_user = 20
    generator.normalize_func = tanh_normalize
    mat, user_list, game_list = generator.construct()  

    result = SVD(mat, feature = Feature, step = Step, Rate = rate)
    rmse = result[2][1]
    csvFile = open("data1.csv", "w")
    writer = csv.writer(csvFile)
    writer.writerow(rmse)
    csvFile.close()