import numpy as np
import pandas as pd

def normalize_z(df):
    dfout = (df - df.mean(axis=0))/df.std(axis=0)
    return dfout

def get_features_targets(df, feature_names, target_names):
    df_feature = df[feature_names]
    df_target = df[target_names]
    try:
        return df_feature.to_frame(), df_target.to_frame()
    except:
        pass
    return df_feature, df_target

def prepare_feature(df_feature):
    shape = df_feature.shape
    return np.concatenate((np.ones((shape[0],1)), df_feature.to_numpy().reshape(shape)), axis = 1)

def prepare_target(df_target):
    shape = df_target.shape
    return df_target.to_numpy().reshape(shape)

def predict(df_feature, beta):
    df_feature = normalize_z(df_feature)
    X = prepare_feature(df_feature)
    return predict_norm(X, beta)

def predict_norm(X, beta):
    res = X @ beta
    for i, v in enumerate(res):
        res[i][0] = max(0, v)
    return res

def add_days(pred,days):
    dataset = pd.DataFrame({'icu_patients': pred[:, 0]})
    dataset['days'] = 0
    dataset['new'] = 'Predicted'
    for i in range(101,101+days):
        dataset['days'][i-101] = i
    return dataset

def split_data(df_feature, df_target, randomize = True, random_state=None, test_size=0.3):
    if random_state:
        np.random.seed(random_state)
        test_indices = np.random.choice(df_feature.index, int(df_feature.shape[0]*test_size), replace=False)
    else:
        test_indices = df_feature.index[:int(df_feature.shape[0]*test_size)]
    train_indices = set(df_feature.index) - set(test_indices)
    df_feature_test = df_feature.loc[test_indices]
    df_feature_train = df_feature.loc[train_indices]
    df_target_test = df_target.loc[test_indices]
    df_target_train = df_target.loc[train_indices]
    return df_feature_train.reset_index(drop = True), df_target_train.reset_index(drop = True), df_feature_test.reset_index(drop = True), df_target_test.reset_index(drop = True)
  
def r2_score(y, ypred):
    ratio = np.matmul((y - ypred).T, (y - ypred)) / np.matmul((y - y.mean()).T, (y - y.mean()))
    return 1 - ratio

def mean_squared_error(target, pred):
    return np.matmul((target - pred).T, (target - pred)) / target.shape[0]

def mean_absolute_error(target,pred):
    y = abs(target-pred)
    return y.sum()/target.shape[0]

def rss(target,pred):
    return np.matmul((target-pred).transpose(),target-pred)

def compute_cost(X, y, beta, l):
    J = 0
    m = X.shape[0]
    pred = np.matmul(X, beta)
    J_reg = (l/(2*m)) * beta.T @ beta
    J = np.matmul((pred - y).T , (pred - y)) * (1/(2*m)) + J_reg
    return J[0][0]

def gradient_descent(X, y, beta, alpha, num_iters, l = 10):
    m = y.shape[0]
    J_storage = []
    for i in range(num_iters):
        pred = np.matmul(X, beta)
        beta -= alpha* (1/m) * (np.matmul(X.T, pred - y) + l * beta)
        J_storage.append(compute_cost(X, y, beta, l))
    return beta, J_storage

def transform_features(df, colname_list, degree, colname_postfix = "_pow"):
    for col in colname_list:
        for exponent in range(2, degree+1):
            colname_exponent = f"{col}{colname_postfix}_{str(exponent)}"
            df[colname_exponent] = df[col] ** exponent
    return df

def n_new_predictions(n, tail, degree, mu, sigma,columns):
    df_feature_extended = pd.concat([tail]*n, axis = 0)
    last_day = tail['days'].iloc[-1]
    new_days = np.arange(last_day+1, last_day+n+1, dtype = int)
    df_feature_extended['days'] = new_days
    df_feature_extended = transform_features(df_feature_extended,columns, degree)
    # display(df_feature_extended)
    df_feature_extended =(df_feature_extended - mu)/sigma
    # display(df_feature_extended)
    return df_feature_extended

def train_model(df_country, feature_colummns, to_transform_features, pred_column, degree = 1, iterations = 2000, alpha = 0.005):
    """
    Trains the model

    Parameters
    ----------
    df_country : DataFrame
        DataFrame of a particular country
    feature_colummns : list
        list of all the column names of the features (must be valid col names in the df)
    to_transform_features : list
        all the features we wish to transform to higher powers
    pred_column : list
        list of one column name that we want to predict
    degree : int
        the exponenent we wish to raise the to_transform_features to (does not include interaction terms)
    iterations : int
        the number of iterations to train the model
    alpha : float
        learning rate

    Returns df_features_copy.tail(1), mu, sigma, beta_final
    -------
    df_features_copy.tail(1)
        The last row of all the features (untransformed).
    mu
        mean of the training+validation set
    sigma
        standard deviation of the training+validation set
    beta_final
        the trained weights for the model (will have 1 extra column. must prepare features before using)
    """
    df_feature, df_target = get_features_targets(df_country,feature_colummns,pred_column)
    df_features_copy = df_feature.copy()
    df_features_transformed = transform_features(df_feature, to_transform_features, degree)
    # split into train+validation and test sets
    df_features_train_val, df_target_train_val, df_features_test, df_target_test = split_data(df_features_transformed, df_target, randomize = True, random_state=100, test_size=0.3)
    # get mean and variance of train+validation set
    mu = df_features_transformed.mean() # or is it df_features_train_val????????
    sigma = df_features_transformed.std()
    df_features_train_val_z = normalize_z(df_features_train_val)
    df_features_train, df_target_train, df_features_val, df_target_val = split_data(df_features_train_val_z, df_target_train_val, randomize = True, random_state=100, test_size=0.3)
    #train on training set and test against validation set for best lambda
    X = prepare_feature(df_features_train)
    target = prepare_target(df_target_train)
    beta = np.zeros((X.shape[1],1))
    mse_list = []
    mae_list = []
    lambda_list = [i*0.02 for i in range(200)]
    for l in lambda_list:  
        beta, J_storage = gradient_descent(X, target, beta, alpha, iterations, l)
        pred = predict(df_features_val, beta)
        mse_list.append(mean_squared_error(df_target_val, pred)[0][0])
        mae_list.append(mean_absolute_error(df_target_val,pred)[0])
    #    print(mean_absolute_error(df_target_val,pred))
    #print(mae_list)
    best_l = lambda_list[mse_list.index(min(mse_list))]
    print(f"{min(mse_list)=},{min(mae_list)=}, {best_l=}")
    # now we retrain the model on the entire training set and test it on the original testing set...
    X_opt = prepare_feature(df_features_train_val_z)
    target_opt = prepare_target(df_target_train_val)
    beta_final, J_storage = gradient_descent(X_opt, target_opt, beta, alpha, iterations, best_l)
    print("beta_final_shape", beta_final.shape)
    print("df_features_test_shape", df_features_test.shape)
    pred = predict(df_features_test, beta_final)
    print(f"MSE: {mean_squared_error(df_target_test, pred)[0][0]}")
    print(f"MAE: {mean_absolute_error(df_target_test, pred)[0]}")
    return df_features_copy.tail(1), mu, sigma, beta_final