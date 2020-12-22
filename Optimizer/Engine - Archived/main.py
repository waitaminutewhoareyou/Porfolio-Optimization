from lsq import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("return.csv",index_col="DATE")
asset_names = df.columns # the first column is date, skip it
rets = df.loc[:, df.columns != 'DATE'].fillna("0%").applymap(lambda x: float(x.split("%")[0])/100)
R_train = rets.iloc[:180,:]
R_test = rets.iloc[180:228, :]
T_train, n = R_train.shape
T_test, n = R_test.shape
P = 12



def Porfolio():
    result = pd.DataFrame(columns=["ret_train", "risk_train", "ret_test", "risk_test", "leverage"])
    V0 = 10

    # Markowitz Optimization
    for target_annual_return in [0.1, 0.2, 0.3, 0.4]:
        rho = target_annual_return / P
        w = train(rho)

        r = R_train @ w  # daily return vector
        leverage = np.abs(w).sum()
        annualized_return_train = float(P * np.mean(r))
        annualized_std_train = float(np.sqrt(P) * np.std(r, ddof=0))

        # print("Annualized return on training set is, ", annualized_return_train)
        # print("Annualized risk on training set is", annualized_std_train)


        r = R_test @ w  # daily return vector
        value_series = V0 * (r + 1).cumprod().values.flatten()
        plt.plot(value_series, label=target_annual_return)
        annualized_return_test = float(P * np.mean(r))
        annualized_std_test = float(np.sqrt(P) * np.std(r, ddof=0))
        # print("Annualized return on test set is, ", annualized_return_test)
        # print("Annualized risk on test set is", annualized_std_test)

        result.loc[target_annual_return] = np.array([annualized_return_train, annualized_std_train,
                                            annualized_return_test, annualized_std_test,
                                            leverage])

        #print("The asset with minimum weight is", asset_names[np.argmin(w)], "when target annual return is",
        #      target_annual_return)

    # Equal Weightage
    w_equal = np.ones((n, 1)) / n
    r = R_train @ w_equal

    leverage = np.abs(w_equal).sum()
    annualized_return_train = float(P * np.mean(r))
    annualized_std_train = float(np.sqrt(P) * np.std(r, ddof=0))
    r = R_test @ w_equal  # daily return vector
    value_series = V0 * (r + 1).cumprod().values.flatten()
    plt.plot(value_series, label="Equal Weightage")
    annualized_return_test = float(P * np.mean(r))
    annualized_std_test = float(np.sqrt(P) * np.std(r, ddof=0))
    result.loc["1/n"] = [annualized_return_train, annualized_std_train,
                         annualized_return_test, annualized_std_test,
                         leverage]

    plt.legend()
    plt.xlabel("Months")
    plt.ylabel("Value")
    plt.show()
    return result

if __name__ == "__main__":
    result = Porfolio()
    result.to_csv("Porfolio Summary.csv")
    pass