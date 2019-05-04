import numpy as np
import matplotlib.pyplot as plt
from data_preparation import *



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

def data_prep(library, games):
    library_set = set(library)
    games_set = set(games)
    prep_user = library_set & games_set
    index = []
    prep_game = []
    for game in games:
        if game in prep_user:
            prep_game.append(game)   #save gameid in order
    for i in range(len(prep_game)):
        for j in range(len(library)):
            if prep_game[i] == library[j]:
                index.append(j)
    return set(prep_game), index 



def user_filter(file_name, library, Lower_limit = 50, User = 1000, played_required = None):
    json_data = open(file_name + ".json", 'r').read()
    user_game_data = json.loads(json_data)
    User_fil = []
    n = 0
    index = []
    for id in user_game_data:
        user_game = user_game_data[id]
        game_list_nonz = []
        game_list = []
        game_nonz = []
        time_nonz = []
        for gameid, time in user_game:
            if not time == 0:
                game_list_nonz.append(gameid)
                game_list.append(gameid)
                game_nonz.append([gameid,time])
                time_nonz.append(time)
            else:
                game_list.append(gameid)
        mean = np.median(time_nonz)
        if played_required:
            useful_game_nonz = data_prep(library, game_list_nonz)
        else:
            useful_game = data_prep(library, game_list)
        len_played = len(useful_game_nonz[0])
        user_game_norm = []
        if played_required:
            if len_played >= Lower_limit:
                index.append(useful_game_nonz[1])
                for game_id in library:
                    if game_id in useful_game_nonz[0]:
                        for gameid, time in game_nonz:
                            if gameid == game_id:
                                user_game_norm.append([gameid, np.tanh(time/2.0/mean)])
                    else:
                        user_game_norm.append([game_id, 0])
                User_fil.append(user_game_norm)    
                n = n+1
        else:
            if len(useful_game[0]) >= Lower_limit:
                index.append(useful_game[1])
                for game_id in library:
                    if game_id in useful_game[0]:
                        for gameid, time in user_game:
                            if gameid == game_id:
                                user_game_norm.append([gameid, np.tanh(time/2.0/mean)])
                    else:
                        user_game_norm.append([game_id, 0])
                User_fil.append(user_game_norm) 
        if n == User:
            break
    if played_required:
        return User_fil, index
    else:
        return User_fil, index


def evaluation(itemfeature, library, Userdata, index, n = 20, feature1 = 20, rate = 0.01):
    #games = []
    #for game_id, time in Userdata:
    #    games.append(game_id)
    #games = [game_id for game_id, time in Userdata]
    #prep_user, index = data_prep(library, games)
    #Prep_userdata = []
    #for game_id, time in Userdata:
    #    if game_id in prep_user:
    #        Prep_userdata.append([game_id, time])
    #print(Prep_userdata)
    Prep_userdata = Userdata
    print(len(Prep_userdata))
    m = 0
    ind = 0
    for game_id, time in Prep_userdata:
        ind = ind + 1
        if not time == 0:
            m = m+1
        if m == n:
            break
    mat_for_learn = [] 
    for game_id, time in Prep_userdata[:ind]:
        mat_for_learn.append(0)
    for game_id, time in Prep_userdata[ind:]:
        mat_for_learn.append(time)

    games_for_learn = np.matrix(mat_for_learn)
    print(len(mat_for_learn))
    
#    games_for_learn = np.matrix([time for gameid, time in Prep_userdata[n : ]])
    rmse_for_learn = []
    a = SVD(games_for_learn, feature = feature1, Rate = rate, Type = 1, ItemFeature = itemfeature)
    rmse_for_learn = a[4]
    userfeature = a[0]
    print(len(userfeature))
    Rmse = 0
    for i in range(n):
        error = np.dot(userfeature, np.matrix(itemfeature[index[i]]).T) - Userdata[index[i]][1]
        Rmse = Rmse + pow(error/n, 2)
    rmse = np.sqrt(Rmse)
    print(rmse)
    return rmse, rmse_for_learn
# end of evaluation part


def SVD_2(mat, game_list, user, user_index, feature = 20, step = 1000, Rate = 0.00001, Type = 0, ItemFeature = [0]):

    # Calculate ARmse step by step

    ARmse = np.inf
    Rmse = 0.0
    lr = Rate
    Lambda = 0.01
    L = [[],[]]
    error = 0.0
    x=0.0
    rmse_for_eva = []
    rmse_for_learn = []
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
                UserFeature[p] = UserFeature[p] + lr * (error * ItemFeature[q] - Lambda * UserFeature[q])
       #type0 for the whole data set, type1 for new userdata.                     
        Rmse = np.sqrt(rmse)
        print("Rmse = ", Rmse, "ARmse = ", ARmse)
        L[0].append(x+i*lr)
        L[1].append(Rmse)
        #
        result = evaluation(ItemFeature, game_list, user, user_index, feature1 = feature)
        rmse_for_eva.append(result[0]) #Calculate error for one step
        rmse_for_learn.append(result[1])
        #
        if Rmse < ARmse:
            ARmse = Rmse
       #     lr = lr*0.99
       #     lr = lr/(1.0 + 0.0001*i)
       # if abs(Rmse)<0.1:
        else:
            break
    LearningProcess = L
    Result = np.dot(UserFeature, ItemFeature.T)
    return UserFeature, ItemFeature, LearningProcess, Result,  ARmse, rmse_for_eva, rmse_for_learn

def main(type = 0, Feature = 20, Step = 300, rate = 0.001):
    file_name = "user_game30k"
    file_name2 = "user_game300k"  # providing data for evaluating
    generator = user_game_matrix(file_name)
# default of "played_required" is True
    generator.played_required = True
    generator.thres_game = None
    generator.thres_user = None
    generator.normalize_func = tanh_normalize
    mat, user_list, game_list = generator.construct()  
    user, index= user_filter(file_name2, game_list, User = 20, played_required = True)
    if type == 0:
        rmse_for_eva = SVD_2(mat, game_list, user[3], index[3], step = Step, Rate = rate) # records of errors step by step
        return rmse_for_eva
    elif type == 1:
        rmse_for_feature = SVD(mat, feature = Feature, Rate = rate) # records of errors for different feature number
        return rmse_for_feature
        

"""    
import json
file_name= "user_game30k"
json_data = open(file_name + ".json", 'r').read()
user_game_data = json.loads(json_data)
"""