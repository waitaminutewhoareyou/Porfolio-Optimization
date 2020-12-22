import pandas as pd
from os.path import dirname, join
project_root = dirname(dirname(__file__))
output_path = join(project_root, 'Result\\DirectStockSelection', "")
from ParallelSelection import Markowitz


df = pd.read_csv(join(project_root, "Data/ret/ret.csv"))
df = df.pivot(index="DATE", columns="PERMNO", values="RET").sort_index()
df.to_csv(join(project_root, "Data/ret/ret_transformed.csv"))

if __name__ == '__main__':
    data = pd.read_csv(join(project_root, "Data/ret/ret_transformed.csv"), index_col="DATE")
    data = data.loc[:, data.columns != 'DATE'].astype(float)
    model = Markowitz(data.iloc[3529:], 0.3)
    #model = Markowitz(data.iloc[4600:], 0.3)
    model.summary(output_path)