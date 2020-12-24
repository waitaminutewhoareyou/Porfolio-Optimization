import pandas as pd
from os.path import dirname, join
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
project_root = dirname(dirname(__file__))
output_path = join(project_root, 'Result\\DirectStockSelection', "")
from StockSelection import Markowitz
from tqdm import tqdm

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
    dictionary = space4mark = {
    'Rho': [0.2, 0.25, 0.3, 0.35, 0.45],
    'kappa': (0, 100),
    'look_back':  [5, 21, 42, 63, 125, 189, 250, 500],
    'rebalancing_frequency':[5, 21]
}
    model = Markowitz(data.iloc[3529:], dictionary)
    model.BayesianHyperOpt()
    # max_iter = 10000000000
    # pbar = tqdm(total=max_iter, desc="Hyperopt")
    # trials = Trials()
    #best = fmin(f, space4mark, algo=tpe.suggest, timeout=57600, trials=trials, verbose=True, show_progressbar=True)
    # pbar.close()
    #print('best:')
    #print(best)
