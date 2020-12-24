import gurobipy as gp
import pandas as pd
import numpy as np


def markowitzSolver(data, w_prev, rho, kappa):
    """
    Perform a single iteration of optimization
    :param data: the matrix where row = date and column = asset names
    :param w_prev: the previously weight vector, if not available then set to []
    :param rho: the normalized return
    :param kappa: cost of trading
    :return: the optimized w
    """

    _, num_assets = data.shape
    assert num_assets <= 520, "Number of assets greater than 520!"
    assets_name = data.columns

    mu = data.mean(axis=0)  # n by 1
    var = data.var(axis=0, ddof=1)

    # Create a new model
    m = gp.Model("portfolio")
    # Add a variable for each stock
    w = pd.Series(m.addVars(assets_name, lb=-0.1, ub=0.1), index=assets_name)
    aux_vars = m.addVars(range(num_assets))
    if not w_prev.size:
        shifting = 0
    else:
        shifting = kappa * gp.quicksum([(w_prev[i] - w[i]) ** 2 for i in range(num_assets)])

    # Objective is to minimize risk (squared).  This is modeled using the
    # covariance matrix, which measures
    # the historical correlation between stocks.
    regularization = gp.quicksum([w[i] ** 2 * var[i] for i in range(num_assets)])  # exclude risk-free cashflow
    portfolio_risk = ((data.dot(w) - rho) ** 2).sum() + regularization + shifting
    m.setObjective(portfolio_risk, gp.GRB.MINIMIZE)

    # Fix budget with a constraint
    # m.addConstr(w.sum() == 1, 'budget')
    m.addConstr(mu.dot(w) == rho, "target return")
    m.addConstr(aux_vars.sum() <= 1, "gross exposure")

    m.addConstrs((aux_vars[i] == gp.abs_(w[i]) for i in range(num_assets)))
    # Optimize model to find the minimum risk portfolio
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 5 * 60)  # allow only 1 minute for each optimization problem
    m.optimize()

    try:
        sol = [v.x for v in m.getVars()]
        w = np.array(sol[:num_assets])  # .reshape((-1, 1))
        w = pd.Series(data=w, index=assets_name[:num_assets])
        leverage = np.sum(np.abs(sol[num_assets:]))
        try:
            assert np.abs(leverage-1) <= 1e-5
            assert np.abs(mu.dot(w) - rho) < 1e-5
        except AssertionError as e:
            print(leverage, mu.dot(w), rho)
    except AttributeError as e:
        w = pd.Series(data=np.zeros(num_assets), index=assets_name[:num_assets])
        print("Infeasible !")

    return w