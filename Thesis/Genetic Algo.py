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
p_crossover = 1 # since p_crossover is equl to 1 there is no need of inserting it into the algorithm, all the genes will undergo crossover
p_mutation = 0.3
p_translation = 0.1
p_swap = 0.1
rep_rate = 0.2
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

def translation(c1, p_translation = 0.3):
    """
    translate all the element of a gene by one
    """
    trans = np.random.rand(1)
    if trans< p_translation:
        c1 = list(c1)
        c1.insert(0,c1.pop())
        c1 = np.array(c1)
    return c1

def swap(c1, p_swap = 0.3):
    """
    swaps the n element of a gene with the n+1
    """
    c1 = list(c1)
    for i in range(len(c1)-1):
        sw = np.random.rand(1)
        if sw < p_swap:
            c1[i], c1[i+1] = c1[i+1], c1[i]
    c1 = np.array(c1)
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

def reproduction(pop, scores, rep_rate = 0.2, p_mutation = 0.3, p_translation = 0.3, p_swap = 0.3):
    """
    create a generation starting form the previous one, using roulette wheel selection and random choice to select the parents
    """
    children = []
    for _ in range(int(len(pop)*rep_rate/2)):
        p_1 = pop[roulette_wheel_selection(scores)]
        p_2 = pop[roulette_wheel_selection(scores)]
        c_1, c_2 = crossover(p_1, p_2)
        c_1, c_2 = mutation(c_1, p_mutation = p_mutation), mutation(c_2, p_mutation = p_mutation)
        c_1, c_2 = translation(c_1, p_translation = p_translation), translation(c_2, p_translation = p_translation)
        c_1, c_2 = swap(c_1, p_swap = p_swap), swap(c_2, p_swap = p_swap)
        children.append(c_1)
        children.append(c_2)
    children = np.array(children)
    return children

def darwin(pop, scores, rep_rate = 0.2):
    """
    removes the worst elements from a population, to make space for the children
    """
    scores = list(scores)
    pop = list(pop)
    for _ in range(int(len(pop)*rep_rate)):
        x = scores.index(sorted(scores)[0])
        pop.pop(x)
        scores.pop(x)
    pop = np.array(pop)
    scores = np.array(scores)
    return pop, scores

def GA(X_train, X_test, y_train, y_test, dropout = 0.2, kernel_size = 2, batch_size = 512, epochs = 2, verbose = 0, p_mutation = 0.3, p_translation = 0.3, p_swap = 0.3, rep_rate = 0.2, pop = 100, gen = 20, n_factors = 84):

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
    scores, gen_best_score, gen_best_set = generation_eval(parents, X_train, X_test, y_train, y_test, dropout = dropout, kernel_size = kernel_size, 
                                                           batch_size = batch_size, epochs = epochs, verbose = verbose)
    if gen_best_score > best_score:
        best_score = gen_best_score
        best_set = gen_best_set
        print(f"Best score gen 1: {best_score}")
    print("Finished generation: 1")   
    for i in range(gen-1):
        children = reproduction(parents, scores, p_mutation = p_mutation, p_translation = p_translation, p_swap = p_swap, rep_rate = rep_rate)
        child_scores, gen_best_score, gen_best_set = generation_eval(children, X_train, X_test, y_train, y_test, dropout = dropout, kernel_size = kernel_size, 
                                                                     batch_size = batch_size, epochs = epochs, verbose = verbose)
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_set = gen_best_set
            print(f"Best score gen {i+2}: {best_score}")
        print(f"Finished generation: {i+2}")
        
        parents, scores = darwin(parents, scores, rep_rate = rep_rate)
        parents = np.concatenate((parents, children))
        scores = np.concatenate((scores, child_scores))
        
    return best_score, best_set


#fix the data to run the GA
X_train, X_test, y_train, y_test = data_fixer(DF, "next")

#run the GA using the parameters defined at the beginning 
best_score, best_set = GA(X_train, X_test, y_train, y_test, rep_rate = rep_rate, p_mutation = p_mutation, pop = pop, gen = gen, n_factors = n_factors)
