import numpy as np
import scipy.sparse as sp
import json
import matplotlib.pyplot as plt

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
        print(file_name + ".json is loaded from the disk\n")
        self.normalize_func = None
        self.played_required = True
        self.thres_user = None
        self.thres_game = None


    def user_game_analysis(self):
        """
        Analyze the statistics of the user_game_data, print the statistics and return two dictionary.
        Return: two dictionaries containing the statistics of the data
            user_stat: {user_id: number of owned games}
            game_stat: {game_id: number of perchase}
        """
        # go throught all data and record two dictionaries
        if self.played_required:
            print("games with playtime 0 is neglected\n")
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
        Use the tow statstics dictionaries to construct the game_list and user_list containing users and games 
        above the thresholds
        """
        # choose a threshold
        if not self.thres_game:
            perchase_median = np.median(list(game_stat.values()))
            self.thres_game = perchase_median
        print("The threshold for game is >", self.thres_game)
        if not self.thres_user:
            owned_game_num_median = np.median(list(user_stat.values()))
            self.thres_user = owned_game_num_median
        print("The threshold for user is >", self.thres_user)
        print()

        # create a temporary matrix for filtering
        user_list = [user for user, num in user_stat.items() if num >= self.thres_user]
        game_list = [game for game, num in game_stat.items() if num >= self.thres_game]
        mat = self.create_matrix(user_list, game_list)

        # iteratively delete users and games that have too few entries (samller than the threshold)
        last_size = np.inf
        eliminated_row = set()
        eliminated_col = set()
        loop_count = 0
        while(last_size > mat.count_nonzero()):
            last_size = mat.count_nonzero()
            # if a user has too few entries, set the whole line to 0
            mat = mat.tocsr()
            for row_ind in range(mat.shape[0]):
                if row_ind not in eliminated_row:
                    num_nonzero_row = mat[row_ind].count_nonzero()
                    if num_nonzero_row <= self.thres_user:
                        mat.data[mat.indptr[row_ind]:mat.indptr[row_ind+1]] = 0.0
                        eliminated_row.add(row_ind)
            # if a game has too few entries, set the whole column to 0
            mat = mat.tocsc()
            for col_ind in range(mat.shape[1]):
                if col_ind not in eliminated_col:
                    num_nonzero_col = mat.getcol(col_ind).count_nonzero()
                    if num_nonzero_col <= self.thres_game:
                        mat.data[mat.indptr[col_ind]:mat.indptr[col_ind+1]] = 0.0
                        eliminated_col.add(col_ind)
            loop_count+=1
        print("The filering went through {} iterations:".format(loop_count))
            
        users_to_delte = {user_list[ind] for ind in eliminated_row}
        user_list = [user for user in user_list if user not in users_to_delte]
        games_to_delete = {game_list[ind] for ind in eliminated_col}
        game_list = [game for game in game_list if game not in games_to_delete]
        print("Number of users after filtering:", len(user_list))
        print("Number of games after filtering:", len(game_list))
        print()
        return user_list, game_list


    def create_matrix(self, user_list, game_list):
        """
        create the sparse matrix with the user_list and game_list
        useful_matrix_data: in the form of  [   [(game_index, time), (game_index, time), (game_index, time),..]      (for id 1)
                                                [(game_index, time), (game_index, time)]                              (for id 2)
                                                [(game_index, time), (game_index, time), (game_index, time),..]      (for id 3)
                                                ...
                                            ]
                each line corrspods to the game of one id, length can be different
        """
        # create useful matrix data only with the games and users in the game_list and user_list
        game_index_dict = {game: index for index, game in enumerate(game_list)}
        useful_matrix_data = []
        for id in user_list:
            game_data =  self.user_game_data[id]
            if self.played_required:
                new_game_data = [(game_index_dict[game], time) for game, time in game_data if time > 0 and game in game_index_dict]
            else:
                new_game_data = [(game_index_dict[game], time) for game, time in game_data if game in game_index_dict]
            useful_matrix_data.append(new_game_data)
        assert(len(useful_matrix_data) == len(user_list))


        num_users = len(user_list)
        num_games = len(game_list)
        # construct matrix
        mat = sp.lil_matrix((num_users, num_games), dtype = np.float64)
        for id_index in range(mat.shape[0]):
            game_data = useful_matrix_data[id_index]
            for game_index, time in game_data:
                mat[id_index, game_index] = time


        assert(mat.shape == (len(user_list), len(game_list)))
        return mat


    def construct(self):
        """
        run the whole process, data analysis and matrix creationg, add the normalization 
        """
        # analysis the statistics of the game data (number of perchase)
        user_stat, game_stat = self.user_game_analysis()
        # remove games and users that have too few perchase
        user_list, game_list = self.user_game_filter(user_stat, game_stat)
        # create the matrix in scipy compressed sparse row format
        mat = self.create_matrix(user_list, game_list)
        if self.normalize_func:
            mat = self.normalize_func(mat)
        else:
            print("No normalization function is specified")
            print()
        return mat, user_list, game_list

        
def plot_normalized(mat, i):
    """
    This can be used to plot the histogram of column i of the matrix.
    """
    col = mat.getcol(i)
    col_nonz = col[col.nonzero()[0]]
    game_line = np.array(col_nonz.todense())
    plt.hist(game_line)
    plt.show()


def tanh_normalize(mat):
    """
    This function determines how the matrix element is calculated from the game time.
    Input:
        mat: empty scipy lil-sparse matrix of the shape (total_num_user, total_num_game)

    """
    # normalization
    # change to csc form for easy column slicing
    mat = mat.tocsc()
    num_col = mat.shape[1]
    for i in range(num_col):
        # calculate the median of one column (for one game)
        col = mat.getcol(i)
        col_nonz = col[col.nonzero()[0]]
        med = np.median(col_nonz.todense(), axis=0)[0,0]
        # normalized with tanh, the median corresponds to tanh(0.5)
        mat[:,i] = (col/2./med).tanh()

    return mat.tolil()

if __name__ == "__main__":
    file_name = "user_game300k"
    generator = user_game_matrix(file_name)
    # default of "played_required" is True
    generator.played_required = True
    generator.thres_game = 20
    generator.thres_user = 20
    generator.normalize_func = tanh_normalize
    mat, user_list, game_list = generator.construct()
    # plot_normalized(mat, 3)
