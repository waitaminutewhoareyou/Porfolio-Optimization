import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import *
import warnings




df = pd.read_csv("return.csv",index_col="DATE")
asset_names = df.columns # the first column is date, skip it
rets = df.loc[:, df.columns != 'DATE'].fillna("0%").applymap(lambda x: float(x.split("%")[0])/100)
R_train = rets.iloc[:180,:]
R_test = rets.iloc[180:228, :]
T_train, n = R_train.shape
T_test, n = R_test.shape
P = 12

# Optimization on training set

def train(rho):

    mu = R_train.mean(axis=0)  # n by 1
    var = R_train.var(axis=0, ddof=0)


    # Create a new model
    m = gp.Model("portfolio")

    # Add a variable for each stock
    w = pd.Series(m.addVars(asset_names, lb=-float('inf')), index=asset_names)
    # Objective is to minimize risk (squared).  This is modeled using the
    # covariance matrix, which measures the historical correlation between stocks.

    regularization = quicksum([w[i]**2 * var[i] for i in range(n-1)]) # exclude risk-free cashflow
    portfolio_risk = ((R_train.dot(w)-rho)**2).sum() + regularization
    m.setObjective(portfolio_risk, GRB.MINIMIZE)

    # Fix budget with a constraint
    m.addConstr(w.sum() == 1, 'budget')
    m.addConstr(mu.dot(w) == rho, "target return")
    # Optimize model to find the minimum risk portfolio
    m.setParam('OutputFlag', 0)
    m.optimize()


    w = np.array([v.x for v in m.getVars()]).reshape((-1,1))
    r = R_train.dot(w).values
    #
    # print("Annualized return on training set is, ", P * r.mean())
    # print("Annualized risk on training set is", np.sqrt(P)*np.std(r, ddof=0))
    return w
