from HyperParameterSearch import AnomalyOpt
import pandas as pd
import numpy as np
import tqdm
import logging
from PeakToTroughAndUnderWater import PeakToTrough, UnderWater
from os.path import dirname, join

project_root = dirname(dirname(__file__))
output_path = join(project_root, 'Result', "")


class NonNaiveEqualWeightage(AnomalyOpt):
    def __init__(self,Data, Rho):
        super().__init__(Data, Rho)

    def weight(self, time_series):
        weight = 1/self.num_assets if np.mean(time_series) >= 0 else -1/self.num_assets
        return weight

    def rolling(self):
        performance_dict = {}  # record the performance of each combination of hyper-parameter
        grid = [(look_back, freq) for look_back in self.look_backs for freq in self.rebalancing_frequency]
        for grid_point in tqdm.tqdm(grid, desc="Rolling", position=0, leave=True):
            look_back, freq = grid_point
            num_periods = self.Data.shape[0]
            W = []  # container of w's
            test_start = look_back  # we start validation once enough data is available

            for test_period in range(test_start, num_periods, freq):
                R_train = self.Data.iloc[max(0, test_period - look_back):test_period, :]
                w = R_train.apply(self.weight, axis=0).dropna()

                np.testing.assert_allclose(w.abs().sum(), 1)

                for _ in range(freq):
                    if len(W) < num_periods - test_start:
                        W.append(w.to_numpy().flatten())
                    else:
                        break

            W = np.array(W)
            R_test = self.Data.iloc[test_start:, :]
            r = np.sum(R_test[self.lag:] * W[:-self.lag], axis=1)  # monthly return vector
            performance_dict[grid_point] = r.values
        return performance_dict

    def run(self):
        performance = self.rolling()  # key is combination of hyper-parameter, value is the return vector
        result_df = pd.Series(performance, name="r").to_frame()  # row index is is combination of hyper-parameter, one column named r cotains the return vector
        pd.DataFrame(result_df["r"].to_list(), index=result_df.index).to_csv(output_path+"return.csv")
        result_df["annual return"] = result_df['r'].apply(lambda x: float(self.P * np.mean(x)))
        result_df["annual risk"] = result_df['r'].apply(lambda x: float(np.sqrt(self.P) * np.std(x, ddof=0)))
        result_df["sharpe ratio"] = result_df["annual return"] / result_df["annual risk"]
        result_df['max_drop_down'] = result_df['r'].apply(PeakToTrough)
        result_df['max_time_under_water '] = result_df['r'].apply(UnderWater)
        # result_df.round(2).to_csv("C:\\Users\\apply\\Desktop\\JYH\\Quant Research\\Porfolio Optimization\\Optimizer\\Result\\normalization_column_3529-4779_with lag_adjusted.csv")
        result_df.round(2).to_csv("C:\\Users\\apply\\Desktop\\JYH\\OneDrive\\Ghost\\Pending\\equal weight.csv")

data = pd.read_csv("C:\\Users\\apply\\Desktop\\JYH\\Quant Research\\Porfolio Optimization\\Optimizer\\Data\\return_2000_2018.csv", index_col="DATE")
data = data.loc[:, data.columns != 'DATE'].fillna("0%").applymap(lambda x: float(x.split("%")[0])/100)

test= NonNaiveEqualWeightage(data[3529:], 0.3)
test.run()
