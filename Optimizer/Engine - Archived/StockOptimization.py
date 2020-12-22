import pandas as pd
from os.path import dirname, join

project_root = dirname(dirname(__file__))
output_path = join(project_root, 'Result', "")

df = pd.read_csv(join(project_root, "Data/ret/ret.csv"))
print(len(df["PERMNO"].unique()))
df = df.pivot(index="DATE", columns="PERMNO", values="RET").sort_index()
df.to_csv(join(project_root, "Data/ret/ret_transformed.csv"))

