import numpy as np
import scipy.sparse as sp
import csv
import json


def user_game_analysis(user_game_data, played_required=True):
    """
    Analyze the statistics of the json data and record two dictionaries containing the statistics of the data
    {id: num of owned game} and {game: num of perchase}
    Input: 
        user_game_data: read from the json file in the form of {"id": [[game1, time1],[game2,time2]...]}
        (optional) thres_user, thres_game: the threshold for filtering the useful data
        (optinoal) played_required: count the game only if it's playtime is larger than 0 
    Return:
        useful_user_list and useful_game_list
    """
    # go throught all data and record two dictionary {id: num of owned game} and {game: num of perchase}
    game_stat = {}
    user_stat = {}
    useful_user_num = 0
    for id in user_game_data:
        game_data = user_game_data[id]
        if not game_data: # profile private
            continue
        owned_game_num = 0
        for game_id, game_time in game_data:
            if (not played_required) or game_time>0:
                owned_game_num += 1
                try:
                    game_stat[game_id] += 1
                except KeyError:
                    game_stat[game_id] = 1
        if owned_game_num > 0:
            useful_user_num +=1
            user_stat[id] = owned_game_num


    # print statistics
    perchase_num_list = list(game_stat.values())
    print("Anaysis of games:")
    print("mean:  ", np.mean(perchase_num_list))
    perchase_median = np.median(perchase_num_list)
    print("median:", perchase_median)
    print("max   :",np.max(perchase_num_list))
    print("number of useful games", len(game_stat))
    print()

    owned_game_num_list = list(user_stat.values())
    print("Anaysis of users:")
    print("mean:  ", np.mean(owned_game_num_list))
    owned_game_num_median = np.median(owned_game_num_list)
    print("median:", owned_game_num_median)
    print("max   :",np.max(owned_game_num_list))
    print("useful user ratio: ", useful_user_num,"/",len(user_game_data))
    print()

    return user_stat, game_stat

def user_game_filter(user_game_data, user_stat, game_stat, thres_user=None, thres_game=None, played_required=True):
    # create useful users and games list
    if not thres_user:
        owned_game_num_median = np.median(list(user_stat.values()))
        thres_user = owned_game_num_median
    print("The threshold for user is", thres_user)
    if not thres_game:
        perchase_median = np.median(list(game_stat.values()))
        thres_game = perchase_median
    print("The threshold for game is", thres_game)
    print()

    useful_user_list = [user for user, num in user_stat.items() if num >= thres_user]
    useful_game_list = [game for game, num in game_stat.items() if num >= thres_game]

    # create useful matrix data with 
    useful_game_set = set(useful_game_list)
    useful_matrix_data = []
    for id in useful_user_list:
        game_data =  user_game_data[id]
        if played_required:
            new_game_data = [(game, time) for game, time in game_data if time > 0 and game in useful_game_set]
        else:
            new_game_data = [(game, time) for game, time in game_data if game in useful_game_set]
        useful_matrix_data.append(new_game_data)
        # It can happen that the filtered user has lesser than thresold number of games because some of games was deleted
        # But we make sure that all games have enough user because this number is smaller


    assert(len(useful_matrix_data) == len(useful_user_list))
    print("Number of users after filtering:", len(useful_user_list))
    print("Number of games after filtering:", len(useful_game_list))
    return useful_matrix_data, useful_user_list, useful_game_list


def matrix_element(mat, id_index, game_data, game_index_dict):
    """
    This function determines how the matrix element is calculated from the game time.
    Weight can be added here if needed
    """
    for game, time in game_data:
        mat[id_index, game_index_dict[game]] = time


def create_matrix(useful_matrix_data, user_list, game_list, played_required=True):
    
    game_index_dict = {game: index for index, game in enumerate(game_list)}
    num_users = len(user_list)
    num_games = len(game_list)
    # construct matrix
    mat = sp.lil_matrix((num_users, num_games), dtype = np.int64)
    for id_index in range(num_users):
        game_data = useful_matrix_data[id_index]

        # construct matrix, weight can be added here later
        matrix_element(mat, id_index, game_data, game_index_dict)

    return mat.tocsr()


file_name = "user_game30000"
# load data
json_data = open(file_name + ".json", 'r').read()
user_game_data = json.loads(json_data)
print(file_name, "is loaded from the disk\n")
# analysis the statistics of the game data (number of perchase)
user_stat, game_stat = user_game_analysis(user_game_data, played_required=True)
# remove games and users that have too few perchase
useful_matrix_data, useful_user_list, useful_game_list = user_game_filter(user_game_data, user_stat, game_stat, thres_user=None, thres_game=None, played_required=True)
# create the matrix in scipy compressed sparse row format
mat= create_matrix(useful_matrix_data, useful_user_list, useful_game_list, played_required=True)

