import pandas as pd
from os.path import dirname, join
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
project_root = dirname(dirname(__file__))
output_path = join(project_root, 'Result\\DirectStockSelection', "")
from StockSelection import Markowitz


df = pd.read_csv(join(project_root, "Data/ret/ret.csv"))
df = df.pivot(index="DATE", columns="PERMNO", values="RET").sort_index()
df.to_csv(join(project_root, "Data/ret/ret_transformed.csv"))

def parallelSearch(data):
    data = data.loc[:, data.columns != 'DATE'].astype(float)
    model = Markowitz(data.iloc[3529:], 0.3)
    # model = Markowitz(data.iloc[4650:], 0.3)
    return model.summary(output_path)


def hyperopt_train_test(params):
    model = Markowitz(data.iloc[4650:], 0.3)
    sharpe_ratio = model.evaluateSharpe(**params)
    return sharpe_ratio

space4mark = {
    'kappa' : hp.uniform('kappa', 0, 100),
    'look_back': hp.choice('look_back', [5, 21, 42, 63, 125, 189, 250, 500]),
    'rebalancing_frequency':hp.choice('rebalancing_frequency', [5, 21])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

if __name__ == '__main__':
    data = pd.read_csv(join(project_root, "Data/ret/ret_transformed.csv"), index_col="DATE")
    # parallelSearch(data)
    best = fmin(f, space4mark, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best:')
    print(best)