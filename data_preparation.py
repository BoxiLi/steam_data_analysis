import numpy as np
import scipy.sparse as sp
import csv
import json

file_name = "user_game30000"
def user_game_analysis(file_name):

    json_data = open(file_name + ".json", 'r').read()
    user_game_data = json.loads(json_data)
    print(file_name, "is loaded from the disk\n")

    game_stat = {}
    user_stat = {}
    userfull_user_num = 0
    for id in user_game_data:
        game_data = user_game_data[id]
        if not game_data:
            continue
        owned_game_num = 0
        for game_id, game_time in game_data:
            owned_game_num += 1
            try:
                game_stat[game_id] += 1
            except KeyError:
                game_stat[game_id] = 1
        userfull_user_num +=1
        user_stat[id] = owned_game_num


    game_list = list(game_stat.keys())
    game_list.sort()

    perchase_num_list = list(game_stat.values())
    print("Anaysis of games:")
    print("mean:  ", np.mean(perchase_num_list))
    perchase_median = np.median(perchase_num_list)
    print("median:", perchase_median)
    print("max   :",np.max(perchase_num_list))
    print()

    owned_game_num_list = list(user_stat.values())
    print("Anaysis of users:")
    print("mean:  ", np.mean(owned_game_num_list))
    owned_game_num_median = np.median(owned_game_num_list)
    print("median:", owned_game_num_median)
    print("max   :",np.max(owned_game_num_list))
    print("useful user ratio: ", userfull_user_num,"/",len(user_game_data))
    print()

    return owned_game_num_median, perchase_median

