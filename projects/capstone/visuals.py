###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(X, x_test_start, y_actual, y_predicted, title):
    
    fig, ax = plt.subplots(figsize = (14,8))
    
    lw = 2
    ax.scatter(X, y_actual, color='darkorange', label='data')
    ax.plot(X, y_predicted, color='navy', lw=lw, label='prediction')
    ax.plot([x_test_start,x_test_start],[min(y_actual),max(y_actual)],color='red', lw=lw) # training test split
    bbox_props = dict(boxstyle="rarrow,pad=0.3",  fc='none', ec="b", lw=2)
    ax.text(x_test_start+10, max(y_actual), "Testing", size=10, bbox=bbox_props)   
    bbox_props = dict(boxstyle="larrow,pad=0.3",  fc='none', ec="b", lw=2)
    ax.text(x_test_start-30, max(y_actual), "Training", size=10, bbox=bbox_props)   
  
    ax.set_xlabel('data')
    ax.set_ylabel('target')
    ax.set_title(title)
    #ax.legend()

def score_predictions(regressor,X,y, title):
    
    scores = []
    
    for index in range(len(X)):
        scores.append(regressor.score(X[:index+1],y[:index+1]))

    
        
    fig, ax = plt.subplots(figsize = (14,8))
    ax.plot(scores)
    ax.set_ylabel('r^2 score')
    ax.set_xlabel('days from training')
    ax.set_title(title)
    
def normalize_data(df):
    return df / df.ix[0, :]

def plot_log(log):
    logdf = pd.DataFrame(log)
    logdf = logdf.rolling(10).mean()
    logdf['q_states'].plot(title="Number of Q-states")
    plt.show()
    logdf['avg_reward'].plot(title="Average reward")
    plt.show()
    logdf['epsilon'].plot(title="epsilon")
    plt.show()


def plot_data(df, title="Stock Prices",label="", xlabel="Dates", ylabel="Price", normalize=False, actions=None):
    '''Plot Stock Prices'''
    if normalize:
        df = normalize_data(df)

    ax = df.plot(title=title, fontsize=20, label=label,figsize = (14,8))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if actions is not None:
        for idx, row in actions.iterrows():
            ax.annotate(row['Action'], xy=(row['Date'],row['Price']),xytext=(0,50),textcoords='offset points',
                arrowprops=dict(arrowstyle="->"))
            
    plt.show()
            
