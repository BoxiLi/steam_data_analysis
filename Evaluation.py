import numpy as np

def user_filter(file_name, Lower_limit = 50, User = 1000):
    json_data = open(file_name + ".json", 'r').read()
    user_game_data = json.loads(json_data)
    User_fil = []
    for id in user_game_data:
        user_game = user_game_data[id]
        if len(user_game) >= Lower_limit:
            User_fil.append(user_game)
    return user_fil
    
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

def evaluation(itemfeature, library, Userdata, n = 20):
    games = [game_id for game_id, time in Userdata]
    prep_user, index = data_prep(library, games)
    Prep_userdata = [[game_id, time] for game_id, time in Userdata if game_id in prep_user]
    games_for_learn = np.matrix(Prep_userdata[n : ])
    a = SVD(games_for_learn, Type = 1, Itemfeature = itemfeature)
    userfeature = a[0]
    Rmse = 0
    for i in range(n):
        error = np.dot(userfeature, np.matrix(itemfeature[index[i]]).T) - Userdata[i][1]
        Rmse = Rmse + pow(error/n, 2)
    rmse = np.sqrt(Rmse)
    return rmse
import json
file_name= "user_game30k"
json_data = open(file_name + ".json", 'r').read()
user_game_data = json.loads(json_data)