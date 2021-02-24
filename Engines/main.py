import pandas as pd
from os.path import dirname, join
project_root = dirname(dirname(__file__))
output_path = join(project_root, 'Result\\DirectStockSelection', "")
from StockSelection import Markowitz


df = pd.read_csv(join(project_root, "Data/ret/ret.csv"))
df = df.pivot(index="DATE", columns="PERMNO", values="RET").sort_index()
df.to_csv(join(project_root, "Data/ret/ret_transformed.csv"))


if __name__ == '__main__':
    # stock
    # data = pd.read_csv(join(project_root, "Data/ret/ret_transformed.csv"), index_col="DATE")

    # parallelSearch(data)

    # anomaly
    # data = pd.read_csv(join(project_root, "Data/Anomaly_clean_2000_2019.csv"), index_col="DATE")
    data = pd.read_csv(join(project_root, "Data/return_2000_2018.csv"), index_col="DATE").applymap(lambda x: float(str(x).strip('%'))/100)
    dictionary = {'Rho': (0.2, 0.5), 'kappa': (0, 100), 'look_back': (5, 500),'rebalancing_frequency': [5, 10, 21]}
    model = Markowitz(data.iloc[3529:], dictionary)
    model.BayesianHyperOpt(200)


    # print(model.evaluateSharpe({'Rho':0.201090702, 'kappa':0.228925325, 'look_back':342, 'rebalancing_frequency':5}))


    # 2.2825760987812727
    # 3.5060462198648144