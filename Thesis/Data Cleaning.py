import numpy as np
import pandas as pd
import datetime

#import the stocks data downloaded from WRDS
DF = pd.read_csv("208fe21cc14c14d3.csv")

#explore it 
print(DF.shape)
print(DF["PERMNO"].nunique())
print(DF.isnull().sum())

#fix the values that don't make sense, volume = to 0 and price <0
DF= DF[DF["VOL"] != 0]
DF = DF[DF["PRC"] >= 0]

#drop duplicates of ticker adn date, it means it's repeated data
DF = DF.drop_duplicates(subset = ["TICKER", "date"])

#converts dates to datetime objects
DF["date"] = DF["date"].astype(str)
DF["date"] = pd.to_datetime(DF["date"])

print(DF["TICKER"].nunique())

#set the index, since it's a time series
DF = DF.set_index(["TICKER", "date"])

def get_SMA(data, column, period):
    """
    function to compute the Simple Moving Average
    """
    SMA = data[column].groupby(level=0, group_keys=False).rolling(period).mean()
    return SMA

def get_EMA(data, column, period):
    """
    function to compute the Exponential Moving Average
    """
    EMA = data[column].unstack(0).ewm(span= period, adjust = False).mean().stack().swaplevel(0, 1).sort_index()
    return EMA

#get the SMA for the 5, 15, and 30 days period
DF["SMA_5"] = get_SMA(DF, "PRC", 5)
DF["SMA_15"] = get_SMA(DF, "PRC", 15)
DF["SMA_30"] = get_SMA(DF, "PRC", 30)

#get the EMA for the 5, 15, and 30 days period
DF["EMA_5"] = get_EMA(DF, "PRC", 5)
DF["EMA_15"] = get_EMA(DF, "PRC", 15)
DF["EMA_30"] = get_EMA(DF, "PRC", 30)

#read the accounting data downloaded from WRDS
accounting = pd.read_csv("Accounting DATA.csv")

#how many companies do we have
print(accounting["TICKER"].nunique())

#drop useless variable 
accounting.drop("datacqtr", axis = 1, inplace = True)

#remove the data you don't need (before 1999)
accounting = accounting[accounting["datafqtr"] != "1998Q4"]

#remove the columns that have too muvh missing information
for col in accounting.columns:
    if accounting[col].isnull().sum() >= 100000:
        accounting = accounting.drop(col, axis = 1)
        
#remove the columns that are simply the same value repeated 
for col in accounting.columns:
    if accounting[col].nunique() == 1:
         accounting = accounting.drop(col, axis = 1)

#set the index
accounting = accounting.set_index(["TICKER", "date"])

#create the variables that you need
accounting["current_ratio"] = accounting["actq"] / accounting["lctq"]
accounting["acidtest_ratio"] = (accounting["actq"] - accounting["invtq"])  / accounting["lctq"]
accounting["cash_ratio"] = accounting["cheq"]  / accounting["lctq"]
accounting["operatingCF_ratio"] = accounting["oancfy"]  / accounting["lctq"]
accounting["debt_ratio"] = accounting["atq"] /accounting["ltq"] 
accounting["D_to_E_ratio"] = accounting["ltq"] /accounting["teqq"] 
accounting["LTD_to_E_ratio"] = accounting["dlttq"] /accounting["teqq"] 
accounting["LTD_to_A_ratio"] = accounting["dlttq"] /accounting["teqq"] 
accounting["interest_c_ratio"] = accounting["oiadpq"] /accounting["xintq"] 
accounting["asset_turnover_ratio"] = accounting["revtq"]/accounting["atq"]
accounting["inventory_turnover_ratio"] = accounting["cogsq"]/accounting["invtq"]
accounting["recievables_turnover_ratio"] = accounting["saleq"]/accounting["rectq"]
accounting["DIO"] = 365/accounting["inventory_turnover_ratio"]
accounting["DSO"] = 365/accounting["recievables_turnover_ratio"]
accounting["gross_margin_ratio"] = (accounting["revtq"]-accounting["cogsq"])/accounting["saleq"]
accounting["operating_margin_ratio"] = accounting["oiadpq"]/accounting["saleq"]
accounting["ROA"] = accounting["niq"]/accounting["atq"]
accounting["ROE"] = accounting["niq"]/accounting["teqq"]
accounting["ROIC"] = accounting["niq"]/accounting["icaptq"]
accounting["book_to_share_ratio"] = accounting["teqq"]/accounting["cshoq"]
accounting["earning_per_share"] = accounting["niq"]/accounting["cshoq"]

#replace costat with dummies
accounting["costat"] = pd.get_dummies(accounting["costat"])

def get_MACD(data, longMA, shortMA):
    """
    function to find the MACD
    """
    MACD = data[shortMA] - data[longMA]
    return MACD

#find the simple MACD and the Exponential MACD
DF["S_MACD"] = get_MACD(DF, "SMA_30", "SMA_5")
DF["E_MACD"] = get_MACD(DF, "EMA_30", "EMA_5")

#get the max and min for 14 days
DF["max14"] = DF["PRC"].groupby(level=0, group_keys=False).rolling(14).min()
DF["min14"] = DF["PRC"].groupby(level=0, group_keys=False).rolling(14).max()

#get the stochastic %K and %D
DF["stochastic_%K"] = (DF["PRC"]-DF["min14"])/(DF["max14"]-DF["min14"])
DF["stochastic_%D"] = get_SMA(DF, "stochastic_%K", 3)

def get_MAX(data, column, period):
    """
    function to get the rolling max of a stacked dataframe
    """
    MAX = data[column].groupby(level=0, group_keys=False).rolling(period).max()
    return MAX

def get_MIN(data, column, period):
    """
    funtion to get the rolling min of an unstacked dataframe
    """
    MIN = data[column].groupby(level=0, group_keys=False).rolling(period).min()
    return MIN

#drop the max and min 14, needed for the stochastic but not included 
#in the final DF
DF.drop(["max14", "min14"], axis = 1, inplace = True)

#get max and min for 5, 15, 30 days
DF["MAX_5"] = get_MAX(DF, "PRC", 5)
DF["MAX_15"] = get_MAX(DF, "PRC", 15)
DF["MAX_30"] = get_MAX(DF, "PRC", 30)

DF["MIN_5"] = get_MIN(DF, "PRC", 5)
DF["MIN_15"] = get_MIN(DF, "PRC", 15)
DF["MIN_30"] = get_MIN(DF, "PRC", 30)

#define the daily true range
DF["true_range"] = DF["PRC"].groupby(level = 0, group_keys = False).diff()

#get the average true range as SMA of the true range
DF["ATR_5"] = get_SMA(DF, "true_range", 5)
DF["ATR_15"] = get_SMA(DF, "true_range", 15)
DF["ATR_30"] = get_SMA(DF, "true_range", 30)

#get the normalized true range
DF["norm_ATR_5"] = DF["ATR_5"]/DF["PRC"]*100
DF["norm_ATR_15"] = DF["ATR_15"]/DF["PRC"]*100
DF["norm_ATR_30"] = DF["ATR_30"]/DF["PRC"]*100

#get the 5, 15, 30 days momemtum
DF["momentum_5"] = DF["PRC"].groupby(level = 0, group_keys = False).diff(5)
DF["momentum_15"] = DF["PRC"].groupby(level = 0, group_keys = False).diff(15)
DF["momentum_30"] = DF["PRC"].groupby(level = 0, group_keys = False).diff(30)

#get the 5, 15, 30 days rate of change
DF["ROC_5"] = DF["PRC"].groupby(level=0, group_keys=False).pct_change(5)
DF["ROC_15"] = DF["PRC"].groupby(level=0, group_keys=False).pct_change(15)
DF["ROC_30"] = DF["PRC"].groupby(level=0, group_keys=False).pct_change(30)

#find the daily return
DF["return"] = DF["PRC"].groupby(level=0, group_keys=False).pct_change()

def computeRSI (data, col, period):
    """
    function to find the Relative Strenght Index 
    """
    diff = data[col].groupby(level=0, group_keys=False).diff(1) # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]

    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=period-1 , min_periods=period).mean()
    down_chg_avg = down_chg.ewm(com=period-1 , min_periods=period).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

#find the 5, 15, 30 days RSI
DF["RSI_5"] = computeRSI(DF,"PRC", 5)
DF["RSI_15"] = computeRSI(DF,"PRC", 15)
DF["RSI_30"] = computeRSI(DF,"PRC", 30)

#find the 5, 15, 30 days standard deviation
DF["std_5"]= DF["return"].groupby(level=0, group_keys=False).rolling(window = 5).std()
DF["std_15"]= DF["return"].groupby(level=0, group_keys=False).rolling(window = 15).std()
DF["std_30"]= DF["return"].groupby(level=0, group_keys=False).rolling(window = 30).std()

#remove the NA values form your DF
DF.dropna(inplace = True)


####### LET'S GO BACK TO ACCOUNTING #######
#drop useless column
accounting.drop("intpny", axis = 1, inplace = True)

#reset the index for accounting
accounting.reset_index(inplace = True)

#identify which tickers are usable in accounting 
usable = []
for tick in accounting["TICKER"].unique():
    d = accounting[accounting["TICKER"] == tick].isna().sum().sum()
    if d <= 200:
        usable.append(tick)
        
#print the number of ussable companies
print(len(usable))

#select in accounting only the valid companies
accounting = accounting[accounting["TICKER"].isin(usable)]

############ TIME TO COMBINE THE DATAFRAMES #########
#select the same companies in DF
DF = DF[DF["TICKER"].isin(usable)]

#make sure that they have the same tickers 
accounting = accounting[accounting["TICKER"].isin(DF["TICKER"].unique())]
#set index
accounting.set_index(["TICKER", "date"], inplace =True)

#impute the null varaibles
def impute_null(df):
    """
    impute the NaN Values in the middle and at the end
    Middle: average of last available value and next available
    End: last available value
    """
    df = df.where(df.notnull(), other=(df.fillna(method='ffill')+df.fillna(method='bfill'))/2)
    if df[len(df)//2 :].isna().any().any() == True:
        df[len(df)//2 :] = df[len(df)//2 :].fillna(method = "ffill")
    return d

#DATA MANIPULATION
acc_trunc = accounting.loc[:, "actq":]
for key in acc_trunc.reset_index()["TICKER"].unique():
    globals()[f'{key}'] = acc_trunc.loc[key]
    
for key in acc_trunc.reset_index()["TICKER"].unique():
    globals()[f'{key}'] = impute_null(globals()[f'{key}'])
    
for key in acc_trunc.reset_index()["TICKER"].unique():
    globals()[f'{key}'] = globals()[f'{key}'].replace(np.nan, 0)
    
for key in acc_trunc.reset_index()["TICKER"].unique():
    globals()[f'{key}']["ticker"] = f"{key}"
    
    
new_acc = pd.DataFrame()
for key in acc_trunc.reset_index()["TICKER"].unique():
    new_acc = new_acc.append(globals()[f'{key}'])
    
new_acc = new_acc.reset_index().set_index(["ticker", "date"])

accounting.loc[:, "actq":] = new_acc

for key in accounting.reset_index()["TICKER"].unique():
    globals()[f'{key}'] = accounting.loc[key]
    
for key in accounting.reset_index()["TICKER"].unique():
    globals()[f'{key}']["datafqtr"] = globals()[f'{key}']["datafqtr"].shift(1)  
    
for key in acc_trunc.reset_index()["TICKER"].unique():
    globals()[f'{key}']["ticker"] = f"{key}"
    
acc = pd.DataFrame()
for key in accounting.reset_index()["TICKER"].unique():
    acc = acc.append(globals()[f'{key}'])
    
    
acc = acc.reset_index().set_index(["ticker", "date"])

#merge the dataframes
DF_1 = pd.merge(DF, sic, on = "TICKER", how = "left")


#transorm the date, to coincide with the accounting quarters 
DF_1["date"] = pd.to_datetime(DF_1["date"])
DF_1["quarter"] = DF_1["date"].dt.quarter
DF_1["quarter"] = "Q" + DF_1["quarter"].astype(str) 
DF_1["quarter"] =DF_1["date"].dt.year.astype(str)+ DF_1["quarter"] 

accounting = accounting[~accounting["datafqtr"].isna()]
DF_1["SICCD"].fillna(0, inplace = True)
DF_1["ticker"] = DF_1["TICKER"]
DF_1["datafqtr"] = DF_1["quarter"]
DF_1.drop(["TICKER", "quarter"], axis = 1, inplace = True)

accounting.drop(["date", "fyearq", "fqtr", "curcdq"], axis = 1, inplace = True)

#remerge the data after fixing the issues
DF_2 = pd.merge(DF_1, accounting, how = "left", on = ["ticker", "datafqtr"])
DF_2.dropna(inplace = True)

#perform feature engineering for the model
DF["datafqtr"] = DF["datafqtr"].astype("category") 
DF["SICCD"] = DF["SICCD"].astype("category") 

for key in DF.reset_index()["ticker"].unique():
    globals()[f'{key}'] = DF.loc[key]
    
for tic in DF.reset_index()["ticker"].unique():
    globals()[f'{tic}']["next"] = globals()[f'{tic}']["PRC"].shift(-1) 
    
for key in DF.reset_index()["ticker"].unique():
    globals()[f'{key}']["ticker"] = f"{key}"
    
DF_2 = pd.DataFrame()
for tic in DF.reset_index()["ticker"].unique():
    DF_2 = DF_2.append(globals()[f'{tic}'])
    
#save to csv and reload

DF_1.to_csv("final_data.csv")
DF = pd.read_csv("final_data.csv")

DF["tic"] = DF["ticker"]
DF = DF.set_index(["ticker", "date"])
DF = DF.drop(["interest_c_ratio", "inventory_turnover_ratio" ], axis = 1)

#feature engineering
DF["SICCD"] = DF["SICCD"].astype("category")
DF["gvkey"] = DF["gvkey"].astype("category") 
DF["datafqtr"] = DF["datafqtr"].astype("category")
DF["tic"] = DF["tic"].astype("category")
DF["PERMNO"] = DF["PERMNO"].astype("category")

DF["SICCD"] = DF["SICCD"].cat.codes
DF["gvkey"] = DF["gvkey"].cat.codes 
DF["datafqtr"] = DF["datafqtr"].cat.codes
DF["PERMNO"] = DF["PERMNO"].cat.codes

DF.drop("COMNAM", axis = 1, inplace = True)