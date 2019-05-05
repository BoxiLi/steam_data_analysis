import numpy as np
from numpy.linalg import inv
from data_preparation import *
from recommender import SVD_recommender


def lsq(A, b):
    """
    Solve x of Ax=b with the least square fit, where len(b) > len(x)
    For new user recoomendation, not used yet.
    """
    C = inv(np.dot(A.T,A)) # this is the corvariance matrix
    return np.dot(C, np.dot(A.T, b))

def eval_pred_error(recommender, hole_indices, hole_values):
    pred_error2 = 0.0
    num_samples = len(hole_indices)
    for i in range(num_samples):
        p,q = hole_indices[i]
        pred = np.dot(recommender.UserFeature[p].T , recommender.ItemFeature[q])
        pred_error2 += (hole_values[i] -pred)**2
    pred_error = np.sqrt(pred_error2/num_samples)
    return pred_error


def dig_hole_evaluation(feature_num = 20, step_num = 10, rate = 0.001, num_samples = 100, file_name = "user_game30k"):
    generator = user_game_matrix(file_name)
    generator.played_required = True
    generator.thres_game = 20
    generator.thres_user = 20
    generator.normalize_func = tanh_normalize
    mat, user_list, game_list = generator.construct()

    # dig holes, pick random coordinates in mat and set them to 0
    row_indices, col_indices = mat.nonzero()
    num_total_entries = len(row_indices)

    pointers = np.random.choice(range(num_total_entries), num_samples, replace=False) # replace = False, no repetition
    hole_indices = list(zip(row_indices[pointers], col_indices[pointers])) # create coordiante pairs [(r1,c1),(r2,c2)...]
    hole_values = [mat[index] for index in hole_indices] # record existing mat entries
    for index in hole_indices:
        mat[index] = 0. # set to 0, delete it from the sparse mat so that it will be fitted)

    # run optimization (see recommender.optimize)
    recommender = SVD_recommender(mat, feature_num)
    learn_procress = [np.inf]*5 # inf just for compare, will be deleted at the end
    eval_process = []
    row_indices, col_indices = recommender.mat.nonzero()
    for i in range(step_num):
        rmse = recommender.svd_step(rate, row_indices, col_indices)
        pred_error = eval_pred_error(recommender, hole_indices, hole_values)
        print("rmse =", rmse, "Eval =", pred_error)
        learn_procress.append(rmse)
        eval_process.append(pred_error)
        if rmse > learn_procress[-5]:
            break
    return recommender.UserFeature, recommender.ItemFeature, learn_procress[5:], eval_process

UserFeature, ItemFeature, learn_procress, eval_process = dig_hole_evaluation(feature_num = 20, step_num = 100, rate = 0.001, num_samples = 100, file_name = "user_game100k")
plt.plot(learn_procress, label = "learn")
plt.plot(eval_process, label = "eval")
plt.show()
plt.legend()
