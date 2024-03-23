import numpy as np 
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statistics as sc 
from scipy.stats import norm


class PortFolio:
    def __init__(self,A,B,C,D,balance):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.balance=balance
        self.datA=0
        self.datB=0
        self.datC=0
        self.datD=0
        self.DataP=0
        self.dataR=0
        self.allocations = np.random.random(4)
        self.allocations /= np.sum(self.allocations)
        
    
    def get_data(self):
        self.datA= yf.download(self.A,period='1y',interval='1d')
        self.datB= yf.download(self.B,period='1y',interval='1d')
        self.datC= yf.download(self.C,period='1y',interval='1d')
        self.datD =yf.download(self.D,period='1y',interval='1d')

    def normalize_returns(self):
        self.datA=self.datA.assign(Normalreturn=self.datA['Close'].values / self.datA.iloc[-len(self.datA['Close'])]['Close'])
        self.datB=self.datB.assign(Normalreturn=self.datB['Close'].values / self.datB.iloc[-len(self.datB['Close'])]['Close'])
        self.datC=self.datC.assign(Normalreturn=self.datC['Close'].values / self.datC.iloc[-len(self.datC['Close'])]['Close'])
        self.datD=self.datD.assign(Normalreturn=self.datD['Close'].values / self.datD.iloc[-len(self.datD['Close'])]['Close'])

    def Set_Allocations(self):
        self.datA= self.datA.assign(Allocations= self.datA['Normalreturn'].values * self.allocations[0])
        self.datB= self.datB.assign(Allocations= self.datB['Normalreturn'].values * self.allocations[1])
        self.datC= self.datC.assign(Allocations= self.datC['Normalreturn'].values * self.allocations[2])
        self.datD= self.datD.assign(Allocations= self.datD['Normalreturn'].values * self.allocations[3])
    
    def Get_Positions(self):
        self.datA= self.datA.assign(Positions= self.datA['Allocations'].values * self.Balance)
        self.datB= self.datB.assign(Positions= self.datB['Allocations'].values * self.Balance)
        self.datC= self.datC.assign(Positions= self.datC['Allocations'].values * self.Balance)
        self.datD= self.datD.assign(Positions= self.datD['Allocations'].values * self.Balance)

    def PortF_View(self):
        all_pos = [self.datA['Positions'],self.datB['Positions'],self.datC['Positions'],self.datD['Positions']]
         
    
        self.DataP['AAPL POS'], self.DataP['TSLA POS'], self.DataP['AMZN POS'], self.DataP['NVDA POS'] = all_pos
        self.DataP['TotalPos'] = self.DataP.sum(axis=1)
    #PLOT PORTFOLIO
        plt.style.use('fivethirtyeight')
        self.DataP['TotalPos'].plot(figsize=(10,8))

    def DPortF_View(self):
        self.DataP.drop('TotalPos', axis=1).plot(figsize=(10,8))
        plt.show()
   
    def Get_SharpeRatio(self):
        self.DataP["Dailyreturn"]= self.DataP["TotalPos"].pct_change(1)
        SharpeRatioDay = self.DataP['Dailyreturn'].mean() / self.DataP["Dailyreturn"].std()
        SharpeRationannual= (252 ** 0.5 )* SharpeRatioDay
        print("Ratio de Sharpe du portefeuille :",SharpeRationannual,"\n")
        return SharpeRationannual

    def Get_SigmaSJ(self):
        AllDreturn=[self.datA['Close'].pct_change(1),self.datB['Close'].pct_change(1),self.datC['Close'].pct_change(1),self.datD['Close'].pct_change(1)]
        self.DataR['AAPLDR'], self.DataR['TSLADR'], self.DataR['AMZNDR'], self.DataR['NVDADR'] = AllDreturn
        self.DataR = self.DataR.cov()
        self.DataR['AAPLDR'] = self.DataR['AAPLDR'] * self.allocations[0]
        self.dataR['TSLADR'] = self.DataR['TSLADR'] * self.allocations[1]
        self.DataR['AMZNDR'] = self.DataR['AMZNDR'] * self.allocations[2]
        self.DataR['NVDADR'] = self.DataR['NVDADR'] * self.allocations[3]
        self.DataR.loc[self.DataR.index[0]]= self.DataR.loc[self.DataR.index[0]]*self.allocations[0]
        self.DataR.loc[self.DataR.index[1]]= self.DataR.loc[self.DataR.index[1]]*self.allocations[1]
        self.DataR.loc[self.DataR.index[2]]= self.DataR.loc[self.DataR.index[2]]*self.allocations[2]
        self.DataR.loc[self.DataR.index[3]]= self.DataR.loc[self.DataR.index[3]]*self.allocations[3]

    def Get_SigmaPF(self):
        K=self.DataR.sum()
        Y=K.sum()
        return Y
    
    def Beta_SousJacents(self):
        pass

    def Beta_PF(self):
        pass
    