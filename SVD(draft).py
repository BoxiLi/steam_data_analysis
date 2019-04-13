import numpy as np
import matplotlib.pyplot as plt
def SVD(feature, step, mat, Rate = 0.01):
    ARmse = 100000000.0
    Rmse = 0.0
    lr = Rate
    Lambda = 0.1
    L = [[],[]]
    error = 0.0
    x=0.0
    
    UserFeature = np.matrix(np.random.rand(mat.shape[0], feature))
    ItemFeature = np.matrix(np.random.rand(mat.shape[1], feature))
    for i in range(step):
        n=0
        rmse = 0.0
        for p in range(mat.shape[0]):
            for q in range(mat.shape[1]):
                if not np.isnan(mat[p,q]):
                    Pred = float(np.dot(UserFeature[p,:], ItemFeature[q,:].T))
                    error = mat[p,q] - Pred
               #     print("error = ",error)
                    rmse = rmse + pow(error, 2)
                #    print("rmse = ",rmse)
                    n = n+1
                    for k in range(feature):
                        UserFeature[p,k] = UserFeature[p,k] + lr * (error * ItemFeature[q,k] - Lambda * UserFeature[p,k])
                        ItemFeature[q,k] = ItemFeature[q,k] + lr * (error * UserFeature[p,k] - Lambda * ItemFeature[q,k])
        Rmse = np.sqrt(rmse/n)
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
    return UserFeature, ItemFeature, LearningProcess, Result
#test
m = np.array([[1.0,2,3,0,0,0,9,8,70],
              [1.0,4,6,3,0,0,9,8,6],
              [8.0,0,0,0,9,6,7,5,0],
              [12.0,5,8,6,3,0,0,0,5],
              [8.0,9,6,4,0,0,0,0,0],
              [1.0,0,0,0,0,0,0,2,5],
              [7.0,8,9,4,5,2,45,8,2],
              [4.0,5,6,0,0,0,0,0,0],
              [12.0,6,8,1,9,6,9.0,5.5,8.7],
              [1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9]])
Result = SVD(9, 20000, m)
