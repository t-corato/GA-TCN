import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LeakyReLU, BatchNormalization, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, explained_variance_score

#impose the initial parameters of the Genetic algorithm
p_crossover = 1 # since p_crossover is equl to 1 there is no need of inseritn it into the algorithm, all the genes will undergo crossover
p_mutation = 0.3
pop = 100
gen = 20
n_factors = 84 #retrieve from size of dataset

#read the dataframe previously create in Data Cleaning.py
DF = pd.read_csv("final_data.csv", index_col = ["ticker", "date"])

#this step is only necessary to speed up the computation, if you have a high computational power or a lot of time, it's not necessary
DF = DF.sample(frac = 0.1)

def crossover(p1, p2): 
    """
    middle point crossover with a random splitpoint
    """
  
   # converting the string to list for performing the crossover 
    l = list(p1) 
    q = list(p2) 
  
    # generating the random number to perform crossover 
    k = random.randint(0, len(l)) 
  
    # interchanging the genes 
    for i in range(k, len(l)): 
        l[i], q[i] = q[i], l[i] 
     
    return np.array(l), np.array(q)

def mutation(c1, p_mutation = 0.3):
    """
    mutation, changing 1 into 0 and the other way around,
    to add more randomness
    """
    flag = np.random.rand(*c1.shape) <= p_mutation
    ind = np.argwhere(flag)
    for i in ind:
        if c1[i] == 0:
            c1[i] = 1
        else:
            c1[i] = 0
    return c1

def roulette_wheel_selection(p):
    """
    selection of a parents using their score
    """
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

def feature_select(X, gene):
    """
    deactivate the columns of the dataframe where the gene is 0
    """
    feature_index = []
    for i in range(len(gene)):
        if gene[i] == 1:
            feature_index.append(i)
    df_filter = X[:, feature_index]
    return df_filter

def get_last(data, target):
    """
    remove the last row of each Ticker and store it to be used in the backtest
    """
    last = {}
    tickers = set(data.index.get_level_values(0))
    for tic in sorted(tickers):
        l = (data.loc[tic][-1:].drop(target, axis = 1)).to_dict(orient = "list")
        last[tic] = l
    last = pd.DataFrame(last).transpose()
    for col in last.columns:
        last[col] = last[col].str[0]
    return last

def data_fixer(DF, target):
    """
    remove the nan from the data and scale it
    """
    last = get_last(DF, target)
    DF = DF.replace([np.inf, -np.inf], np.nan)
    DF = DF.dropna()
    X = DF.drop([target], axis = 1)
    y = DF[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def mean_absolute_percentage_error(y_true, y_pred):
    """
    define the metric that will be used to evaluate the fitness of each cromosome
    """
    y_pred = y_pred.reshape(y_pred.shape[0])
    return np.mean((np.abs(y_true - y_pred)) / y_true) * 100

def evaluate(X_train, X_test, y_train, y_test, gene, dropout = 0.2, kernel_size = 2, batch_size = 512, epochs = 2, verbose = 0):
    """
    evaluate a cromosome using the TCN
    """
    X_filt= feature_select(X_train, gene)
    X_test_filt= feature_select(X_test, gene)
    X_filt = np.expand_dims(X_filt, axis=2)
    X_test_filt = np.expand_dims(X_test_filt, axis=2)
    model = Sequential()
    model.add(Conv1D(32, kernel_size, padding = "causal", input_shape = X_filt.shape[1:]))
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
    
    model.compile(loss= "mean_squared_error", optimizer= "adam")
    
    model.fit(x=X_filt, y=y_train, batch_size = batch_size, epochs=epochs, validation_data=(X_test_filt, y_test), verbose=verbose)
    
    pred = model.predict(X_test_filt)
    perc_err = mean_absolute_percentage_error(y_test, pred)
    score = (100 - perc_err)/100
    return score

def generation_eval(pop, X_train, X_test, y_train, y_test, dropout = 0.2, kernel_size = 2, batch_size = 512, epochs = 2, verbose = 0):
    """
    evaluate all the scores of a generation, returns all the scores, the best score and the gene that gave the best score
    """
    scores = []
    best_score = 0
    best_set = []
    for i in range(len(pop)):
        score = evaluate(X_train, X_test, y_train, y_test, pop[i], dropout = dropout, kernel_size = kernel_size, batch_size = batch_size, epochs = epochs, verbose = verbose) 
        scores.append(score)
        if score > best_score:
            best_score = score
            best_set = pop[i]
    scores = np.array(scores)
    return scores, best_score, best_set

def reproduction(pop, scores, p_mutation = 0.3):
    """
    create a generation starting form the previous one, using roulette wheel selection and random choice to select the parents
    """
    children = []
    for _ in range(int(len(pop)/2)):
        p_1 = pop[roulette_wheel_selection(scores)]
        p_2 = random.choice(pop)
        c_1, c_2 = crossover(p_1, p_2)
        c_1, c_2 = mutation(c_1, p_mutation = p_mutation), mutation(c_2, p_mutation = p_mutation)
        children.append(c_1)
        children.append(c_2)
    children = np.array(children)
    return children

def GA(X_train, X_test, y_train, y_test, dropout = 0.2, kernel_size = 2, batch_size = 512, epochs = 2, verbose = 0, p_mutation = 0.3, pop = 100, gen = 20, n_factors = 84):
    """
    run the genetic algorithm for n generation with m genes, storing the best score and the best gene
    """
    parents = []
    for i in range(pop):
        i = np.random.choice([0, 1], size=(n_factors,), p=[1./3, 2./3])
        parents.append(i)
    parents = np.array(parents) 
    
    best_score = 0
    best_set = []
    for i in range(gen):
        scores, gen_best_score, gen_best_set = generation_eval(parents, X_train, X_test, y_train, y_test, dropout = dropout, kernel_size = kernel_size, 
                                                   batch_size = batch_size, epochs = epochs, verbose = verbose)
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_set = gen_best_set
            print(f"Best score gen {i+1}: {best_score}")
        
        children = reproduction(parents, scores, p_mutation = p_mutation)
        parents = children
        
    return best_score, best_set


#fix the data to run the GA
X_train, X_test, y_train, y_test = data_fixer(DF, "next")

#run the GA using the parameters defined at the beginning 
best_score, best_set = GA(X_train, X_test, y_train, y_test, p_mutation = p_mutation, pop = pop, gen = gen, n_factors = n_factors)