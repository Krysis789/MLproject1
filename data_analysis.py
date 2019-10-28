import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from proj1_helpers import *
DATA_TRAIN_PATH = '../../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

def histograms(tX):
    # Plot histogram each column to look at data distribution for all features
    # (with raw data and after removing -999 representing missing values)
    plt.subplots(len(tX[0]),2, figsize=(20,300))
    for i in range(len(tX[0])):
        
        plt.subplot(len(tX[0]),2,2*i+1)
        plt.hist(tX[:,i], bins = 50, log=True)
        plt.title("column " + str(i))
        plt.ylim(0.5, 1000000)
        
        x = tX[:,i]
        x = [value for value in x if value != -999]
        plt.subplot(len(tX[0]),2,2*i+2)
        plt.hist(x, bins=50, log=True)
        plt.title("column " + str(i) + " without -999")
        plt.ylim(0.5, 1000000)
        
    plt.show()

data = pd.read_csv(DATA_TRAIN_PATH)
data.columns = range(32)
data = data[range(2,32)]
data.head()

def linear_correlation(data)
    # Compute correlation matrix for all 
    corr = pd.DataFrame.corr(data)
    corr["to ignore"]=[-1]*30
    corr.style.background_gradient(cmap='coolwarm',axis=1).set_precision(2)
    
    data.replace(to_replace=-999, value= np.nan, inplace=True)
    corr = pd.DataFrame.corr(data)
    corr["to ignore"]=[-1]*30
    corr.style.background_gradient(cmap='coolwarm',axis=1).set_precision(2)


# Pairplot enable to look for pairwise relationship between features
def pair_plot(data):
    sns.set(style="ticks", color_codes=True)
    sns.pairplot(data)