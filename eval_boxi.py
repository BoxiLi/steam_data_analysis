import numpy as np
from numpy.linalg import inv
from data_preparation import *
from recommender import SVD_recommender
from multiprocessing import Pool


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
    pred_list = np.empty(num_samples)
    for i in range(num_samples):
        p,q = hole_indices[i]
        pred_list[i] = recommender.mu + recommender.UserUncertain[p] + recommender.ItemUncertain[q] + np.dot(recommender.UserFeature[p].T, recommender.ItemFeature[q])
    
    pred_error2 = np.sum(np.square(hole_values - pred_list))
    pred_error = np.sqrt(pred_error2/(num_samples-1))
    corr = np.corrcoef(pred_list, hole_values)[0,1]
    return pred_error, corr


def dig_hole_evaluation(feature_num = 20, step_num = 10, rate = 0.001, num_samples = 300, regu = 0.01, file_name = "user_game30k", seed = None):
    if seed is not None:
        np.random.seed(seed)
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
    hole_values = np.array([mat[index] for index in hole_indices]) # record existing mat entries
    for index in hole_indices:
        mat[index] = 0. # set to 0, delete it from the sparse mat so that it will be fitted)

    # run optimization (see recommender.optimize)
    recommender = SVD_recommender(mat, feature_num, regu)
    learn_procress = [np.inf]*5 # inf just for compare, will be deleted at the end
    eval_process = []
    corr_list = []
    row_indices, col_indices = recommender.mat.nonzero()
    for i in range(step_num):
        rmse = recommender.svd_step(rate, row_indices, col_indices)
        pred_error, corr = eval_pred_error(recommender, hole_indices, hole_values)
        # print("rmse =", rmse, "Eval =", pred_error)
        learn_procress.append(rmse)
        eval_process.append(pred_error)
        corr_list.append(corr)
        if i%50==0:
            print("{} steps finished".format(i))
        # if rmse > learn_procress[-5]:
        #     break
    return learn_procress[5:], eval_process, corr_list


# def test_process(regu):
#     return dig_hole_evaluation(feature_num = 20, step_num = 3000, rate = 0.001, num_samples = 100, regu = regu, file_name = "user_game30k", seed = 6)


# if __name__ =="__main__":
#     regu_list = np.linspace(0.00, 0.1, 6)
#     process_num = min(len(regu_list), 11)
#     with Pool(process_num) as p:
#         result = p.map(test_process, regu_list)

#     fig1 = plt.figure(dpi=200)
#     ax1 = fig1.subplots()
#     fig2 = plt.figure(dpi=200)
#     ax2 = fig2.subplots()
#     fig3 = plt.figure(dpi=200)
#     ax3 = fig3.subplots()
#     for i in range(len(regu_list)):
#         learn_procress = result[i][0]
#         eval_process = result[i][1]
#         corr = result[i][2]
#         ax1.plot(learn_procress, label = "regu {}".format(regu_list[i]))
#         ax2.plot(eval_process, label = "regu {}".format(regu_list[i]))
#         ax3.plot(corr, label = "regu {}".format(regu_list[i]))
#     ax1.set_xlabel("step")
#     ax2.set_xlabel("step")
#     ax3.set_xlabel("step")
#     ax1.set_ylabel("Loss")
#     ax2.set_ylabel("prediction error")
#     ax3.set_ylabel("correlation")
#     ax1.legend()
#     ax2.legend()
#     ax3.legend()
#     ax1.set_title("Loss function for different regularization")
#     ax2.set_title("Evaluation for different regularization")
#     ax3.set_title("Correlation for different regularization")
#     plt.show()


def test_process(feature_num):
    return dig_hole_evaluation(feature_num, step_num = 3000, rate = 0.001, num_samples = 1000, regu = 0.01, file_name = "user_game300k", seed=300)

if __name__ =="__main__":
    # feature_num_list = np.linspace(2, 10, 5).astype(int)
    feature_num_list = [1,3,5,7,9,11]
    process_num = min(len(feature_num_list), 11)


    with Pool(process_num) as p:
        result = p.map(test_process, feature_num_list)


    fig1 = plt.figure(dpi=200)
    ax1 = fig1.subplots()
    fig2 = plt.figure(dpi=200)
    ax2 = fig2.subplots()
    fig3 = plt.figure(dpi=200)
    ax3 = fig3.subplots()
    for i in range(len(feature_num_list)):
        learn_procress = result[i][0]
        eval_process = result[i][1]
        corr = result[i][2]
        ax1.plot(learn_procress, label = "F {}".format(feature_num_list[i]))
        ax2.plot(eval_process, label = "F {}".format(feature_num_list[i]))
        ax3.plot(corr, label = "F {}".format(feature_num_list[i]))
    ax1.set_xlabel("step")
    ax2.set_xlabel("step")
    ax3.set_xlabel("step")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("prediction error")
    ax3.set_ylabel("correlation") 
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_title("Loss function for different feature number")
    ax2.set_title("Evaluation for different feature number")
    ax3.set_title("Correlation for different feature number")
    plt.show()
