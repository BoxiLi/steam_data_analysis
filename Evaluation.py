def data_prep(library, games):
    library_set = set(library)
    games_set = set(games)
    prep_user = list(library_set & games_set)
    index = []
    for i in range(len(prep_user)):
        for j in range(len(library)):
            if prep_user[i] == library[j]:
                index.append(j)
    return prep_user, index

def evaluation(itemfeature, library, Userdata, n = 20):
    Prep_userdata = []
    for game_id in Userdata:
        try:
            Prep_userdata.append(library[game_id])
        except KeyError:
            continue
    prep_user, index = data_prep(library, games)
    games_for_eval = Prep_userdata[ : n]
    games_for_learn = Prep_userdata[n : ]
    a = SVD(games_for_learn, Type = 1, Itemfeature = itemfeature)
    userfeature = a[0]
    Rmse = 0
    for i in range(n):
        error = np.dot(userfeature, np.matrix(itemfeature[index[i]]).T) - Userdata[i]
        Rmse = Rmse + pow(error/n, 2)
    rmse = np.sqrt(Rmse)
    return rmse