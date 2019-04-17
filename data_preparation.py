import numpy as np
import scipy.sparse as sp
import csv
import json

class user_game_matrix(object):
    def __init__(self, file_name):
        """
        Data type:
        user_game_data: read from the json file in the form of {id1: [[game1, time1],[game2,time2]...], id2:...}
        played_required: bool, if true, game with playtime 0 will be neglected
        thres_user: users with games fewer than thres_user will be neglected. If set None, median will be used
        thres_user: games with perchase fewer than thres_game will be neglected. If set None, median will be used
        user_list: a list of users, it will be updated by the filter with the corresponding threshold
        game_list: a list of games, it will be updated by the filter with the corresponding threshold
        """
        json_data = open(file_name + ".json", 'r').read()
        self.user_game_data = json.loads(json_data)
        print(file_name, "is loaded from the disk\n")
        self.played_required = True
        self.thres_user = None
        self.thres_game = None
        self.user_list = None
        self.game_list = None


    def user_game_analysis(self):
        """
        Analyze the statistics of the user_game_data, print the statistics and return two dictionary.
        Return: two dictionaries containing the statistics of the data
            user_stat: {user_id: number of owned games}
            game_stat: {game_id: number of perchase}
        """
        # go throught all data and record two dictionary {id: num of owned game} and {game: num of perchase}
        if self.played_required:
            print("games with playtime 0 is neglected")
        game_stat = {}
        user_stat = {}
        useful_user_num = 0
        for id in self.user_game_data:
            game_data = self.user_game_data[id]
            if not game_data: # profile private
                continue
            owned_game_num = 0
            for game_id, game_time in game_data:
                if (not self.played_required) or game_time>0:
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
        print("useful user ratio: ", useful_user_num,"/",len(self.user_game_data))
        print()

        return user_stat, game_stat

    def user_game_filter(self, user_stat, game_stat):
        """
        Use the tow statstics dictionaries to filter the game_list and user_list, the corresponding threshold is used and 
        user_list, game_list are updated
        Return:
            useful_matrix_data: a list of game_list [[game1,game2,game3...], [game1, game2...]...]
                                useful_matrix_data[i] is the games owned by user_list[i]
        """
        # create useful users and games list
        if not self.thres_game:
            perchase_median = np.median(list(game_stat.values()))
            self.thres_game = perchase_median
        print("The threshold for game is", self.thres_game)
        if not self.thres_user:
            owned_game_num_median = np.median(list(user_stat.values()))
            self.thres_user = owned_game_num_median
        print("The threshold for user is", self.thres_user)
        print()

        self.user_list = [user for user, num in user_stat.items() if num >= self.thres_user]
        self.game_list = [game for game, num in game_stat.items() if num >= self.thres_game]

        # create useful matrix data with 
        useful_game_set = set(self.game_list)
        useful_matrix_data = []
        for id in self.user_list:
            game_data =  self.user_game_data[id]
            if self.played_required:
                new_game_data = [(game, time) for game, time in game_data if time > 0 and game in useful_game_set]
            else:
                new_game_data = [(game, time) for game, time in game_data if game in useful_game_set]
            useful_matrix_data.append(new_game_data)
            # It can happen that the filtered user has lesser than thresold number of games because some of games was deleted
            # But we make sure that all games have enough user because this number is smaller


        assert(len(useful_matrix_data) == len(self.user_list))
        print("Number of users after filtering:", len(self.user_list))
        print("Number of games after filtering:", len(self.game_list))
        return useful_matrix_data


    def create_matrix(self, useful_matrix_data):
        """
        create the desired matrix with "matrix_func"
        """
        game_index_dict = {game: index for index, game in enumerate(self.game_list)}
        num_users = len(self.user_list)
        num_games = len(self.game_list)
        # construct matrix
        mat = sp.lil_matrix((num_users, num_games), dtype = np.float64)
        matrix_func(self.user_list, self.game_list, mat, useful_matrix_data, game_index_dict)

        assert(mat.shape == (len(self.user_list), len(self.game_list)))
        return mat.tocsr(), self.user_list, self.game_list


    def construct(self, matrix_element):
        """
        run the whole process, data analysis and matrix construction
        """
        # analysis the statistics of the game data (number of perchase)
        user_stat, game_stat = self.user_game_analysis()
        # remove games and users that have too few perchase
        useful_matrix_data = self.user_game_filter(user_stat, game_stat)
        # create the matrix in scipy compressed sparse row format
        return self.create_matrix(useful_matrix_data)


def matrix_func(user_list, game_list, mat, useful_matrix_data, game_index_dict):
    """
    This function determines how the matrix element is calculated from the game time.
    Input:
        user_list: a list of user id
        game_list: a list of game id
        mat: scipy lil-sparse matrix of the shape (total_num_user, total_num_game)
        game_index_dict: a dictionary {game_id: index of the matrix column}
    """
    for id_index in range(len(user_list)):
        game_data = useful_matrix_data[id_index]
        for game, time in game_data:
            mat[id_index, game_index_dict[game]] = time
        

file_name = "user_game"
generator = user_game_matrix(file_name)
# default of "played_required" is True
generator.played_required = False
generator.thres_game = None
generator.thres_user = None
mat, user_list, game_list = generator.construct(matrix_func)
