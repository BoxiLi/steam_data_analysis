import numpy as np
import matplotlib.pyplot as plt
def SVD(mat, feature = 20, step = 20000, Rate = 0.01, Type = 0, ItemFeature = [0]):
    ARmse = 100000000.0
    Rmse = 0.0
    lr = Rate
    Lambda = 0.1
    L = [[],[]]
    error = 0.0
    x=0.0
    
    UserFeature = np.matrix(np.random.rand(mat.shape[0], feature))
    if Type == 0:
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
                #type0 for the whole data set, type1 for new userdata.  
                    if Type == 0:
                        for k in range(feature):
                            UserFeature[p,k] = UserFeature[p,k] + lr * (error * ItemFeature[q,k] - Lambda * UserFeature[p,k])
                            ItemFeature[q,k] = ItemFeature[q,k] + lr * (error * UserFeature[p,k] - Lambda * ItemFeature[q,k])
                    else if Type == 1:
                        for k in range(feature):
                            UserFeature[p,k] = UserFeature[p,k] + lr * (error * ItemFeature[q,k] - Lambda * UserFeature[p,k])
       #type0 for the whole data set, type1 for new userdata.                     
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
        if np.isnan(User[i]):
            PItem.append(ItemFeature[i])
    Result = np.dot(UserFeature, PItem.T)
    Rec = np.sort(Result)
    Index = np.argsort(Result)
    n = 0
    for i in range(RN):
        n = n+1
        if Rec[i] < User:
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

m = np.array([[1.0,np.nan,3,np.nan,0,np.nan,9,8,70],
              [1.0,4,6,3,0,0,9,8,6],
              [8.0,np.nan,0,np.nan,9,6,7,5,0],
              [12.0,5,8,6,3,0,np.nan,0,5],
              [8.0,9,6,4,0,np.nan,0,0,0],
              [1.0,np.nan,0,np.nan,0,0,0,2,5],
              [7.0,8,9,4,5,2,45,8,2],
              [4.0,5,6,0,np.nan,0,np.nan,0,0],
              [12.0,6,8,1,9,6,9.0,5.5,8.7],
              [1.1,2.2,3.3,np.nan,5.5,6.6,7.7,8.8,9.9]])
Result = SVD(m, feature = 9, step = 20000)
