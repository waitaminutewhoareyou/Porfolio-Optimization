import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import *
import numpy.testing as npt
import warnings
import tqdm
df = pd.read_csv("return.csv",index_col="DATE")
asset_names = df.columns # the first column is date, skip it
num_periods = df.shape[0] #228 months
rets = df.loc[:, df.columns != 'DATE'].fillna("0%").applymap(lambda x: float(x.split("%")[0])/100)
R_train = rets.iloc[:180,:]
R_test = rets.iloc[180:228, :]
T_train, n = R_train.shape
T_test, n = R_test.shape
P = 12

# Optimization on training set

def train(R_train, w_old= np.zeros((n,1)), rho=0.3, kappa=0):

    mu = R_train.mean(axis=0)  # n by 1
    var = R_train.var(axis=0, ddof=1)


    # Create a new model
    m = gp.Model("portfolio")

    # Add a variable for each stock
    w = pd.Series(m.addVars(asset_names, lb=-float('inf')), index=asset_names)

    if w_old is None:
        shifting = 0
    else:
        shifting = kappa*quicksum([(w_old[i] - var[i])**2 for i in range(n)])

    # Objective is to minimize risk (squared).  This is modeled using the
    # covariance matrix, which measures the historical correlation between stocks.
    regularization = quicksum([w[i]**2 * var[i] for i in range(n)])  # exclude risk-free cashflow
    portfolio_risk = ((R_train.dot(w)-rho)**2).sum() + regularization + shifting
    m.setObjective(portfolio_risk, GRB.MINIMIZE)


    # Fix budget with a constraint
    m.addConstr(w.sum() == 1, 'budget')
    m.addConstr(mu.dot(w) == rho, "target return")
    # Optimize model to find the minimum risk portfolio
    m.setParam('OutputFlag', 0)
    m.optimize()

    w = np.array([v.x for v in m.getVars()]).reshape((-1,1))
    #r = R_train.dot(w).values
    return w

output = pd.DataFrame(columns=["return", "risk"])
#for Kappa in tqdm([100/(10**i) for i in range(0,5)], desc="Kappa iteration", position=2, leave=True):
for Rho in tqdm.tqdm([0.2, 0.3], desc="Rho"):
    lookback_period = num_periods
    W = []  # container of w's
    test_start = 122  # we start testing from t=180
    for test_period in tqdm.tqdm(range(test_start, num_periods), desc="test_period", position=0, leave=True):

        R_train = rets.iloc[max(0, test_period-lookback_period):test_period, :]
        w_old = W[-1] if len(W) != 0 else None

        w = train(R_train, w_old, rho= Rho/P)
        W.append(w.flatten())

    W = np.array(W)
    np.testing.assert_array_almost_equal(W.sum(axis=1), np.ones_like(W.sum(axis=1)), decimal=2)

    V0 = 10
    R_test = rets.iloc[test_start:, :]
    r = np.sum(R_test * W, axis=1)  # daily return vector
    value_series = V0 * (r + 1).cumprod().values.flatten()
    plt.plot(value_series, label=f"Rho={Rho}")
    annualized_return_test = float(P * np.mean(r))
    annualized_std_test = float(np.sqrt(P) * np.std(r, ddof=0))
    print("Annualized return on test set is, ", annualized_return_test, "for target return", Rho)
    print("Annualized risk on test set is", annualized_std_test,"for target return", Rho)
    output.loc[Rho] = [annualized_return_test, annualized_std_test]
    pd.DataFrame(data=W, index=R_test.index, columns=asset_names).to_csv(f"W for Rho = {Rho}.csv")
# Equal Weightage
w_equal = np.ones((n, 1)) / n


leverage = np.abs(w_equal).sum()
R_test = rets.iloc[test_start:, :]
r = R_test @ w_equal  # daily return vector
V0 = 10
value_series = V0 * (r + 1).cumprod().values.flatten()
plt.plot(value_series, label="Equal Weightage")
annualized_return_test = float(P * np.mean(r))
annualized_std_test = float(np.sqrt(P) * np.std(r, ddof=0))
output.loc["1/n"] = [annualized_return_test, annualized_std_test]
plt.legend()
plt.xlabel("Months")
plt.ylabel("Value")
plt.show()
output.to_csv("Time Varying Result.csv")
