# Recommender part 
        
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import *


def pow_normalize(mat):
    return mat/60**(1./3.)


def SVD(mat, feature = 20, step = 1000, Rate = 0.0000001, Type = 0, ItemFeature = [0]):

    ARmse = np.inf
    Rmse = 0.0
    lr = Rate
    Lambda = 0.01
    L = [[],[]]
    error = 0.0
    x=0.0
    UserFeature = np.matrix(np.random.rand(mat.shape[0], feature))
    if Type == 0:
        ItemFeature = np.matrix(np.random.rand(mat.shape[1], feature))
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
            rmse = rmse + pow(error/useful_entry_number, 2)
            #type0 for the whole data set, type1 for new userdata.  
            if Type == 0:
                UserFeature[p] = UserFeature[p] + lr * (error * ItemFeature[q] - Lambda * UserFeature[p])
                ItemFeature[q] = ItemFeature[q] + lr * (error * UserFeature[p] - Lambda * ItemFeature[q])
            elif (Type == 1):
                UserFeature[p] = UserFeature[p] + lr * (error * ItemFeature[q] - Lambda * UserFeature[p])
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
    Result = np.dot(UserFeature, ItemFeature.T)
    return UserFeature, ItemFeature, LearningProcess, Result


def ItemBase(OwnItem, ItemFeature, NNN):
    Corr = []
    for i in range(ItemFeature.shape[0]):
        Corr.append(np.corrcoef(OwnItem, ItemFeature[i])[0][1])
    SCorr = np.sort(Corr)
    b = np.argsort(Corr)
    n = 0
    for i in range(ItemFeature.shape[0]):
        n = n+1
        if SCorr[i+1] < 0.5 or (i+1) >= NNN:
            break
    return  b[1:n+1]

#return n recommendation


def UserSVD(User, ItemFeature0, RN):
    Avg = np.average(User)
    UserFeature = SVD(User, Type = 1, ItemFeature = ItemFeature0)[0]
    PItem = []
    for i in range(len(User)):
        if np.isnan(User[i]):
            PItem.append(ItemFeature[i])
    Result = np.dot(UserFeature, PItem.T)
    Rec = np.sort(Result)
    Index = np.argsort(Result)
    n = 0
    for i in range(RN):
        n = n+1
        if Rec[i] < User:
            break
    return Index[:n]
    

def Recommed(User, ItemFeature):
    a = np.argsort(User)
    Rec = []
    if (len(User) - np.isnan(User).sum()) < 50:
        for i in range(5):
            Rec.append(ItemBase(ItemFeature[a[i],:], ItemFeature, 5))
    else:
        Rec.append(UserSVD(User, ItemFeature, 25))



file_name = "user_game100k"
generator = user_game_matrix(file_name)
# default of "played_required" is True
generator.played_required = None
generator.thres_game = None
generator.thres_user = None
generator.normalize_func = pow_normalize
mat, user_list, game_list = generator.construct()
result = SVD(mat)
