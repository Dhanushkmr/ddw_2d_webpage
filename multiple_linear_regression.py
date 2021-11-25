import numpy as np
import pandas as pd
import math

def normalize_z(df):
    dfout = df.copy()
    dfout = (df - df.mean(axis=0)) / df.std(axis=0)
    return dfout


def get_features_targets(df, feature_names, target_names):
    df_feature = df[feature_names]
    df_target = df[target_names]
    return df_feature, df_target

def prepare_feature(df_feature): # X matrix: # rows -> # samples, # cols -> # features
    cols = len(df_feature.columns)
    feature = df_feature.to_numpy().reshape(-1, cols)
    X = np.concatenate((np.ones((feature.shape[0],1)), feature), axis=1)
    return X


def prepare_target(df_target):
    cols = len(df_target.columns)
    target = df_target.to_numpy().reshape(-1, cols)
    return target
    pass

def split_data(df_feature, df_target, random_state=None, test_size=0.5):
    indices = df_feature.index
    if random_state != None:
        np.random.seed(random_state)
    k = int(test_size * len(indices))
    if test_size != 0:
        test_index = np.random.choice(indices,k,replace=False)
        indices = set(indices)
        test_index = set(test_index)
        train_index = indices - test_index
        df_feature_train = df_feature.loc[train_index, :]
        df_feature_test = df_feature.loc[test_index, :]
        df_target_train = df_target.loc[train_index, :]
        df_target_test = df_target.loc[test_index, :]
        # df_test_names = df_target_test['country']
    else:
        df_feature_train = df_feature
        df_target_train = df_target
        df_feature_test = 0
        df_target_test = 0
    # df_target_train.drop('country',axis = 1)
    # df_target_test.drop('country',axis = 1)
    # df_feature_train.drop('country',axis = 1)
    # df_feature_test.drop('country',axis = 1)
    return df_feature_train, df_feature_test, df_target_train, df_target_test
def compute_cost(X, y, beta):
    J = 0
    m = X.shape[0] #get the number of rows in data frame X
    error = np.matmul(X, beta) - y
    error_sq = np.matmul(error.T, error)
    J = (1/(2*m)) * error_sq #matrix
    J = J[0][0] #to get scalar value
    return J

def gradient_descent(X, y, beta, alpha, num_iters):
    J_storage = []
    m = y.shape[0]
    print(y.shape)
    for i in range(num_iters):
        cost = compute_cost(X,y,beta)
        right = np.matmul(X,beta)
        transpose_x = np.transpose(X)
        beta = beta - (alpha/(m))*np.matmul(transpose_x,(right - y))
        J_storage.append(cost)
    return beta, J_storage

def predict(df_feature, beta):
    new_X = normalize_z(df_feature)
    new_X = prepare_feature(new_X)
    return predict_norm(new_X,beta)
    
def predict_norm(X, beta):
    df = np.matmul(X,beta)
    return df

def r2_score(y, ypred):
    new_y = y-ypred
    transpose_y  = np.transpose(new_y)
    SSres = np.matmul(transpose_y,new_y)
    mean = y.mean()
    ss_y = y - mean
    ss_t = np.transpose(ss_y)
    SStot = np.matmul(ss_t,ss_y)
    print(1-SSres/SStot)
    return 1-SSres/SStot

def mean_squared_error(target, pred):
    n = target.shape[0]
    minus = target-pred
    minus_t = np.transpose(minus)
    return (1/n)*np.matmul(minus_t,minus)
    

def mean_absolute_error(target,pred):
    y = abs(target-pred)
    return y.sum()/target.shape[0]

def transform_features(df, colname_list, degree, colname_postfix = "_pow"):
    for col in colname_list:
        if col != 'country':
            for exponent in range(2, degree+1):
                colname_exponent = f"{col}{colname_postfix}_{str(exponent)}"
                df[colname_exponent] = df[col] ** exponent
                df[colname_exponent] = np.where(df[col]!=0, np.log(df[col] ** exponent),0)
            df[col] = np.where(df[col]>0, np.log(df[col]),0)
    return df

def train_model(df_continent,feature_column,target_column,degree=1,test_size=0.3,alpha = 0.01,iterations = 1500):
    print("real_mean", df_continent.new_cases.mean())
    df_features, df_target = get_features_targets(df_continent,feature_column,target_column)
    df_features_transformed = transform_features(df_features, feature_column, degree)
    df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features_transformed,df_target,100 ,test_size)
    # Normalize using z normalization
    mu = df_features.mean()
    sig = df_features.std()
    df_features_train_z = normalize_z(df_features_train)
    # print(df_features_train_z)
    X = prepare_feature(df_features_train_z)
    target = prepare_target(df_target_train)
    beta = np.zeros((X.shape[1],1))
    beta, J_storage = gradient_descent(X, target, beta, alpha, iterations)
    if test_size == 0:
        pred = predict(df_features,beta)
        # plt.plot(J_storage)
        mse = mean_squared_error(df_target.to_numpy(),pred)
        mae = mean_absolute_error(df_target.to_numpy(),pred)
        print("RMSE:")
        print(math.sqrt(mse))
        print("MAE:")
        print(mae)
        print("Beta:")
        print(beta)
    if test_size != 0:
        pred = predict(df_features_test,beta)
        # plt.plot(J_storage)
        mse = mean_squared_error(df_target_test.to_numpy(),pred)
        mae = mean_absolute_error(df_target_test.to_numpy(),pred)
        print("RMSE:",math.sqrt(mse) )
        print("MAE:", mae)
        print("beta:", beta)
        print("mu:", mu)
        print("sig:", sig)
    return beta, mu, sig