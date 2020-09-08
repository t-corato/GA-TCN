import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LeakyReLU, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler

print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

DF = pd.read_csv("final_data.csv", index_col = ["ticker", "date"])

def get_last(data, target):
    last = {}
    tickers = set(data.index.get_level_values(0))
    for tic in sorted(tickers):
        l = (data.loc[tic][-1:].drop(target, axis = 1)).to_dict(orient = "list")
        last[tic] = l
    last = pd.DataFrame(last).transpose()
    for col in last.columns:
        last[col] = last[col].str[0]
    return last

def data_fixer(DF, target, test_size = 0.2):
    DF = DF.replace([np.inf, -np.inf], np.nan)
    DF = DF.dropna()
    X = DF.drop([target], axis = 1)
    y = DF[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=101)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

#select only the portion of the data we need 
def data_selector(data, ym_start = "2003-01", ym_end= "2020-03"):
    df = data.reset_index()
    df = df[(df["date"] >= ym_start) & (df["date"]<= ym_end)]
    df = df.set_index(["ticker", "date"])
    return df

def feature_select(X, gene):
    feature_index = []
    for i in range(len(gene)):
        if gene[i] == 1:
            feature_index.append(i)
    df_filter = X[:, feature_index]
    return df_filter

#predict the first non available period 
def predict_next(last, model):
    pred_reg_next = model.predict(last)
    return pred_reg_next

#combine the predictions with the rest of the data
def combiner(ticks, pred_reg, last_prices):
    pred_reg = np.reshape(pred_reg, pred_reg.shape[0])
    data = pd.concat([pd.Series(sorted(ticks)), pd.Series(pred_reg), last_prices], axis = 1, keys = ["tickers", "pred", "last_price"])
    return data

#select the companies to invest in 
def selector(data):
    select = data[data["pred"]>data["last_price"]]
    if len(select)>10:
        select["diff"] = (select["pred"] - select["last_price"])/select["last_price"]
        select = select.sort_values(by = "diff", ascending= False)[:10]
    return select

#compute the return of the first non available period
def next_returns(data, select, ym_end = "2019-12-31"):
    tic = select["tickers"]
    df = data.reset_index()
    date = str(np.datetime64(ym_end) + np.array(1, 'timedelta64[D]'))
    df = df[(df["ticker"].isin(tic)) & (df["date"] == date)]
    ret = df["return"]
    if len(select) == 0:
        weights = 1
    else:
        weights = 1/len(select)
    partial_ret = ret*weights 
    total_ret = sum(partial_ret)
    return total_ret, partial_ret, tic

def backtesting(data, target, best_set = None, period = 14, test_size = 0.2, ym_start = "2016-01-01", ym_end= "2019-12-31", kernel_size = 2, dropout = 0.2, epochs = 10, batch = 1024, verbose = 0):
    
    #handle the data
    df_1 = data_selector(data, ym_start = ym_start, ym_end= str(np.datetime64(ym_end) - np.array(period, 'timedelta64[D]')))
    X_train, X_test, y_train, y_test = data_fixer(df_1, target, test_size= test_size)
    X_train = feature_select(X_train, best_set)
    X_test = feature_select(X_test, best_set)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    #define the model
    model = Sequential()
    model.add(Conv1D(32, kernel_size, padding = "causal", input_shape = X_train.shape[1:]))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Conv1D(64, kernel_size, padding = "causal",  dilation_rate = 2))
    model.add(LeakyReLU(alpha = 0.01))
    model.add(BatchNormalization())

    model.add(Conv1D(128, kernel_size, padding = "causal", activation = "relu", dilation_rate = 3))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation = "relu"))
    

    #compile and train the model
    model.compile(loss= "mean_squared_error", optimizer= "adam")
    model.fit(x=X_train, y=y_train, batch_size = batch,
              epochs=epochs, validation_data=(X_test, y_test), verbose=verbose)
  
    #predict next period returns, period of [period] days
    total_rets = {}
    partial_rets = {}
    tickers = {}
    for i in range(1, period+1):
        try:
            date = str(np.datetime64(ym_end) - np.array(period, 'timedelta64[D]')+ np.array(i, 'timedelta64[D]'))
            df = data_selector(data, ym_start = ym_start, ym_end=  date)
            last = get_last(df, target)
            tics = last.index
            last_prices = last.reset_index()["PRC"]
            last = last.replace([np.inf, -np.inf], np.nan)
            last = last.dropna()
            scaler = MinMaxScaler()
            last = scaler.fit_transform(last)
            last = feature_select(last, best_set)
            last = np.expand_dims(last, axis = 2)
            pred_reg_next = predict_next(last, model)
            combined = combiner(tics, pred_reg_next, last_prices)
            selected = selector(combined)
            total_ret, partial_ret, tic = next_returns(data, selected, ym_end= date)
            total_rets[date] = total_ret
            partial_rets[date] = partial_ret
            tickers[date] = tic
        except:
            pass
    
    #define the total strategy results
    result = []
    for key in total_rets.keys():
        backtest_result = total_rets[key]
        result.append(1+backtest_result)
    total_res = np.cumprod(result)

    return total_res, result, tickers, total_rets, partial_rets

#import the best set from the GA
best_set = pd.read_csv("best_set.csv")
best_set = best_set["gene"]
best_set = np.array(best_set)

#backtest on a certain amount of data for a certain period
total_res, result, tics, total_rets, partial_rets = backtesting(DF, "next", best_set = best_set, period = 14, test_size = 0.2, ym_start = "1999-01-01", ym_end= "2019-11-01", kernel_size = 2, dropout = 0.2, epochs = 2, batch = 512, verbose = 1)