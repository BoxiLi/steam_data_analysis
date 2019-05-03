import numpy as np
import matplotlib.pyplot as plt
from data_preparation import *



def pow_normalize(mat):
    return mat/60**(1./3.)


def SVD(mat, feature = 20, step = 1000, Rate = 0.00001, Type = 0, ItemFeature = [0]):
    
    ARmse = np.inf
    Rmse = 0.0
    lr = Rate
    Lambda = 0.01
    L = [[],[]]
    error = 0.0
    x=0.0
    np.random.seed(0)
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
    return UserFeature, ItemFeature, LearningProcess, Result,  ARmse




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
        if User[i]==0:
            PItem.append(ItemFeature0[i])
    Result = np.dot(UserFeature, np.matrix(PItem).T)
    Rec = np.sort(Result)
    Index = np.argsort(Result)
    n = 0
    for i in range(RN):
        n = n+1
        if Rec[i] < Avg:
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


"""
file_name = "user_game30k"
generator = user_game_matrix(file_name)
# default of "played_required" is True
generator.played_required = None
generator.thres_game = None
generator.thres_user = None
generator.normalize_func = tanh_normalize
mat, user_list, game_list = generator.construct()
result = SVD(mat)
"""


# This part for Evaluation

def user_filter(file_name, Lower_limit = 50, User = 1000):
    json_data = open(file_name + ".json", 'r').read()
    user_game_data = json.loads(json_data)
    User_fil = []
    n = 0
    for id in user_game_data:
        user_game = user_game_data[id]
        game_nonz = []
        for gameid, time in user_game:
            if not time == 0:
                game_nonz.append(time)    
        mean = np.mean(game_nonz)
        user_game_norm = []
        if len(user_game) >= Lower_limit:
            for gameid, time in user_game:
                user_game_norm.append([gameid, np.tanh(time/2.0/mean)])
            User_fil.append(user_game_norm)
            n = n+1
        if n == User:
            break
    return User_fil

def data_prep(library, games):
    library_set = set(library)
    games_set = set(games)
    prep_user = list(library_set & games_set)
    index = []
    for i in range(len(prep_user)):
        for j in range(len(library)):
            if prep_user[i] == library[j]:
                index.append(j)
    return set(prep_user), index 

def evaluation(itemfeature, library, Userdata, n = 20, feature1 = 20, rate = 0.1):
    #games = []
    #for game_id, time in Userdata:
    #    games.append(game_id)
    games = [game_id for game_id, time in Userdata]
    prep_user, index = data_prep(library, games)
    #Prep_userdata = []
    #for game_id, time in Userdata:
    #    if game_id in prep_user:
    #        Prep_userdata.append([game_id, time])
    #print(Prep_userdata)
    Prep_userdata = [[game_id, time] for game_id, time in Userdata if game_id in prep_user]
    games_for_learn = np.matrix([time for gameid, time in Prep_userdata[n : ]])
    print(games_for_learn)
    a = SVD(games_for_learn, feature = feature1, Rate = rate, Type = 1, ItemFeature = itemfeature)
    userfeature = a[0]
    Rmse = 0
    for i in range(n):
        error = np.dot(userfeature, np.matrix(itemfeature[index[i]]).T) - Userdata[i][1]
        Rmse = Rmse + pow(error/n, 2)
    rmse = np.sqrt(Rmse)
    return rmse
# end of evaluation part


def SVD_2(mat, game_list, user, feature = 20, step = 1000, Rate = 0.00001, Type = 0, ItemFeature = [0]):

    # Calculate ARmse step by step

    ARmse = np.inf
    Rmse = 0.0
    lr = Rate
    Lambda = 0.01
    L = [[],[]]
    error = 0.0
    x=0.0
    rmse_for_eva = []
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
        rmse_for_eva.append(evaluation(ItemFeature, game_list, user))

        if Rmse < ARmse:
            ARmse = Rmse
       #     lr = lr*0.99
       #     lr = lr/(1.0 + 0.0001*i)
       # if abs(Rmse)<0.1:
        else:
            break
    LearningProcess = L
    Result = np.dot(UserFeature, ItemFeature.T)
    return UserFeature, ItemFeature, LearningProcess, Result,  ARmse, rmse_for_eva

def main(type = 0):
    file_name = "user_game30k"
    file_name2 = "user_game300k"  # providing data for evaluating
    generator = user_game_matrix(file_name)
# default of "played_required" is True
    generator.played_required = None
    generator.thres_game = None
    generator.thres_user = None
    generator.normalize_func = tanh_normalize
    mat, user_list, game_list = generator.construct()  
    user = user_filter(file_name2, User = 20)
    if type == 0:
        rmse_for_eva = SVD_2(mat, game_list, user, Rate = 0.001) # records of errors step by step
    elif type == 1:
        rmse_for_feature = SVD(mat, feature = 19, Rate = 0.001) # records of errors for different feature number
        

"""    
import json
file_name= "user_game30k"
json_data = open(file_name + ".json", 'r').read()
user_game_data = json.loads(json_data)
"""