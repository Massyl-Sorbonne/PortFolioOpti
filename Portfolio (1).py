import numpy as np 
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statistics as sc 
from scipy.stats import norm

Balance = 1000000
def PortfolioView(A, B, C, D,Balance):
    TicketA = yf.Ticker(A)
    TicketB = yf.Ticker(B)
    TicketC = yf.Ticker(C)
    TicketD = yf.Ticker(D)
  # ETUDE DE A 
    global dataA
    dataA = yf.download(A, start='2022-06-02', interval="1d")

    # NORMALIZE RETURNS
    dataA= dataA.assign(Normalreturn=dataA['Close'].values / dataA.iloc[-len(dataA['Close'])]['Close'])
    

    # Add random allocation
    global allocations
    allocations = np.random.random(4)
    allocations /= np.sum(allocations)  # Divide by the sum to ensure the total is 1
    #ALLOCATION A
    dataA= dataA.assign(Allocations= dataA['Normalreturn'].values * allocations[0])

    #ADD POSITION 
    dataA= dataA.assign(Positions= dataA['Allocations'].values * Balance)
  
 # ETUDE DE B
    global dataB
    dataB = yf.download(B, start='2022-06-02', interval="1d")

    # NORMALIZE RETURNS
    dataB= dataB.assign(Normalreturn=dataB['Close'].values / dataB.iloc[-len(dataB['Close'])]['Close'])
    

    #ALLOCATION B
    dataB= dataB.assign(Allocations= dataB['Normalreturn'].values * allocations[1])

    #ADD POSITION 
    dataB= dataB.assign(Positions= dataB['Allocations'].values * Balance)
    
 # ETUDE DE C
    global dataC
    dataC = yf.download(C, start='2022-06-02', interval="1d")

    # NORMALIZE RETURNS
    dataC= dataC.assign(Normalreturn=dataC['Close'].values / dataC.iloc[-len(dataC['Close'])]['Close'])
    

    #ALLOCATION C
    dataC= dataC.assign(Allocations= dataC['Normalreturn'].values * allocations[2])

    #ADD POSITION 
    dataC= dataC.assign(Positions= dataC['Allocations'].values * Balance)


     # ETUDE DE D
    global dataD
    dataD = yf.download(D, start='2022-06-02', interval="1d")

    # NORMALIZE RETURNS
    dataD= dataD.assign(Normalreturn=dataD['Close'].values / dataD.iloc[-len(dataD['Close'])]['Close'])
    

    #ALLOCATION D
    dataD= dataD.assign(Allocations= dataD['Normalreturn'].values * allocations[3])

    #ADD POSITION 
    dataD= dataD.assign(Positions= dataD['Allocations'].values * Balance)

 #PORTFOLIO
    all_pos = [dataA['Positions'],dataB['Positions'],dataC['Positions'],dataD['Positions']]
    global DataP 
    DataP= pd.DataFrame()
    DataP['AAPL POS'], DataP['TSLA POS'], DataP['AMZN POS'], DataP['NVDA POS'] = all_pos
    DataP['TotalPos'] = DataP.sum(axis=1)
    #PLOT PORTFOLIO
    plt.style.use('fivethirtyeight')
    DataP['TotalPos'].plot(figsize=(10,8))


    # EACH SOUS JACENTS 
    DataP.drop('TotalPos', axis=1).plot(figsize=(10,8))
    plt.show()
    return Balance


#FCT 1 APPEL 
PortfolioView('AAPL', 'TSLA', 'AMZN', 'NVDA',Balance)

def SharpeRation():
       # SHARPE RATIO CALCULUS
    DataP["Dailyreturn"]= DataP["TotalPos"].pct_change(1)
    SharpeRatioDay = DataP['Dailyreturn'].mean() / DataP["Dailyreturn"].std()
    SharpeRationannual= (252 ** 0.5 )* SharpeRatioDay
    print("Ratio de Sharpe du portefeuille :",SharpeRationannual,"\n")
    return SharpeRationannual
#FCT 2 APPL 
SharpeRation()


#DATAR = TABLEAU CLOSE/ ADJ RETURN.percetenge change

def var_covPortfolio():
    global DataR
    DataR=pd.DataFrame()
    AllDreturn=[dataA['Close'].pct_change(1),dataB['Close'].pct_change(1),dataC['Close'].pct_change(1),dataD['Close'].pct_change(1)]
    DataR['AAPLDR'], DataR['TSLADR'], DataR['AMZNDR'], DataR['NVDADR'] = AllDreturn
    DataR = DataR.cov()
    DataR['AAPLDR'] = DataR['AAPLDR'] * allocations[0]
    DataR['TSLADR'] = DataR['TSLADR'] * allocations[1]
    DataR['AMZNDR'] = DataR['AMZNDR'] * allocations[2]
    DataR['NVDADR'] = DataR['NVDADR'] * allocations[3]
    DataR.loc[DataR.index[0]]= DataR.loc[DataR.index[0]]*allocations[0]
    DataR.loc[DataR.index[1]]= DataR.loc[DataR.index[1]]*allocations[1]
    DataR.loc[DataR.index[2]]= DataR.loc[DataR.index[2]]*allocations[2]
    DataR.loc[DataR.index[3]]= DataR.loc[DataR.index[3]]*allocations[3]

    print ('Matrice De Var-Cov du PortFolio : \n',DataR)
    return DataR
var_covPortfolio()

def V_H():
   K=DataR.sum()
   Y=K.sum()
   print(Y)
   return Y
V_H()

def BetaA():
    Index = yf.download('VOO', start='2022-06-02', interval="1d")
    DataBeta= pd.DataFrame({'AAPL Returns': dataA['Close'].pct_change().dropna(),'Index Returns': Index['Close'].pct_change().dropna()})
    DataBeta.dropna(inplace=True)
    # Séparation des variables indépendantes (X) et dépendantes (Y)
    X = DataBeta['Index Returns'].values.reshape(-1, 1)
    Y = DataBeta['AAPL Returns'].values

# Création du modèle de régression linéaire
    model = LinearRegression()
# Entraînement du modèle
    model.fit(X, Y)
# Obtention du coefficient de régression (bêta)
    beta_aapl = model.coef_[0]
    print('Beta du sous jacents AAPL',beta_aapl)
    return beta_aapl
def BetaB():
    Index = yf.download('SPY', start='2022-06-02', interval="1d")
    DataBeta= pd.DataFrame({'AAPL Returns': dataB['Close'].pct_change().dropna(),'Index Returns': Index['Close'].pct_change().dropna()})
    DataBeta.dropna(inplace=True)
    # Séparation des variables indépendantes (X) et dépendantes (Y)
    X = DataBeta['Index Returns'].values.reshape(-1, 1)
    Y = DataBeta['AAPL Returns'].values

# Création du modèle de régression linéaire
    model = LinearRegression()
# Entraînement du modèle
    model.fit(X, Y)
# Obtention du coefficient de régression (bêta)
    beta_aapl = model.coef_[0]
    print('Beta du sous jacents TSLA',beta_aapl)
    return beta_aapl
def BetaC():
    Index = yf.download('SPY', start='2022-06-02', interval="1d")
    DataBeta= pd.DataFrame({'AAPL Returns': dataC['Close'].pct_change().dropna(),'Index Returns': Index['Close'].pct_change().dropna()})
    DataBeta.dropna(inplace=True)
    # Séparation des variables indépendantes (X) et dépendantes (Y)
    X = DataBeta['Index Returns'].values.reshape(-1, 1)
    Y = DataBeta['AAPL Returns'].values

# Création du modèle de régression linéaire
    model = LinearRegression()
# Entraînement du modèle
    model.fit(X, Y)
# Obtention du coefficient de régression (bêta)
    beta_aapl = model.coef_[0]
    print('Beta du sous jacents TSLA',beta_aapl)
    return beta_aapl

def BetaD():
    Index = yf.download('QQQ', start='2022-06-02', interval="1d")
    DataBeta= pd.DataFrame({'AAPL Returns': dataD['Close'].pct_change().dropna(),'Index Returns': Index['Close'].pct_change().dropna()})
    DataBeta.dropna(inplace=True)
    # Séparation des variables indépendantes (X) et dépendantes (Y)
    X = DataBeta['Index Returns'].values.reshape(-1, 1)
    Y = DataBeta['AAPL Returns'].values

# Création du modèle de régression linéaire
    model = LinearRegression()
# Entraînement du modèle
    model.fit(X, Y)
# Obtention du coefficient de régression (bêta)
    beta_aapl = model.coef_[0]
    print('Beta du sous jacents TSLA',beta_aapl)
    return beta_aapl

def betaportefeuille():
    b_p= BetaA()*allocations[0] + BetaB()*allocations[1] + BetaC()*allocations[2] + BetaD()*allocations[3] 
    print("Beta Portefeuille : ", b_p)
    return b_p
betaportefeuille()
#Calcul du drift à completer plsu atrd 
def DriftCal():
    drift= DataP['TotalPos'].mean()
    pass
def RandomWalk():
    a = np.random.normal(DriftCal(),V_H(),1)


