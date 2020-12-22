import sys
import os
from os.path import dirname, join
project_root = dirname(dirname(__file__))
solver_path = join(project_root, 'Solvers')
sys.path.insert(1, solver_path)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import tqdm
from PeakToTroughAndUnderWater import PeakToTrough, UnderWater
from MarkowitzSolver import markowitzSolver
from matplotlib import pyplot as plt
from multiprocessing import Pool


class Markowitz:

    def __init__(self, Data, Rho):
        self.Data = Data
        self.asset_names = self.Data.columns  # remember to drop the column ['DATE']
        self.num_assets = len(self.Data.columns)
        self.kappas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        self.look_backs = [5, 21, 42, 63, 125, 189, 250, 500]  # [7, 30, 90, 180] #number of days to look back
        self.rebalancing_frequency = [5, 21]
        self.P = 250  # monthly frequency
        self.Rho = Rho  # target annualized
        self.rho = self.Rho / self.P
        self.lag = 1

    def getPorfolioUniverse(self, df):
        return df.dropna().index

    def normalizeFactors(self, data):
        """
        Scale each column to have the same annualized std. but the scaled factor is capped by 2.
        :param data: the training data
        :return: Normalized data
        """
        normalization_factors = data.apply(lambda x: min(0.15 / np.sqrt(self.P) / x.std(), 2), axis=0)
        data = data.apply(lambda x: min(0.15 / np.sqrt(self.P) / x.std(), 2) * x, axis=0)
        return data, normalization_factors

    def preProcessing(self, test_period, look_back, num_periods):
        porfolio_universe = self.getPorfolioUniverse(self.Data.iloc[min(test_period + self.lag, num_periods - 1), :])  # Select all investable assets
        R_train = self.Data[porfolio_universe]
        R_train = R_train.iloc[max(0, test_period - look_back):test_period, :]
        R_train = R_train.loc[:, R_train.isnull().sum() < 0.8 * R_train.shape[0]].fillna(0)  # Total number of NaN entries in a column must be less than 80% of total entries:

        R_train_normalized, normalization_factors = self.normalizeFactors(R_train)
        tradeable_universe = R_train_normalized.columns

        return porfolio_universe, R_train_normalized, normalization_factors

    def solve(self, solver, *args):
        return solver(*args)


    def postProcess(self, partial_w, normalization_factors, num_periods, test_start, freq):
        # unpack the weight vectors and sigma factor
        w = pd.Series(index=self.asset_names)
        w[partial_w.index] = partial_w.values

        sigma_factor = pd.Series(index=self.asset_names)
        sigma_factor[normalization_factors.index] = normalization_factors

        for _ in range(freq):
            if W.shape[0] < num_periods - test_start:
                W = W.append(w, ignore_index=True).fillna(0)
                sigma_factors = sigma_factors.append(sigma_factor, ignore_index=True).fillna(0)

            else:
                break

    @ property
    def hyperParameterGrid(self):
        grid = [(kappa, look_back, freq) for kappa in self.kappas for look_back in self.look_backs for freq in
                self.rebalancing_frequency]
        return grid

    def evaluate(self, grid_point):

        kappa, look_back, freq = grid_point
        num_periods = self.Data.shape[0]
        W = pd.DataFrame(columns=self.asset_names)  # container of w's
        sigma_factors = pd.DataFrame(columns=self.asset_names)  # container of normalization factors
        test_start = look_back  # we start validation once enough data is available

        for test_period in tqdm.tqdm(range(test_start, num_periods, freq),desc="Processed"):

            porfolio_universe, R_train_normalized, normalization_factors = self.preProcessing(test_period, look_back,
                                                                                              num_periods)

            w_old = np.array([]) if W.empty else W[porfolio_universe].iloc[-1].fillna(0).to_numpy()
            save_stdout = sys.stdout
            sys.stdout = open('trash', 'w')
            partial_w = self.solve(markowitzSolver, R_train_normalized, w_old, self.rho, kappa)
            sys.stdout = save_stdout

            # unpack the weight vectors and sigma factor
            w = pd.Series(index=self.asset_names)
            w[partial_w.index] = partial_w.values

            sigma_factor = pd.Series(index=self.asset_names)
            sigma_factor[normalization_factors.index] = normalization_factors

            for _ in range(freq):
                if W.shape[0] < num_periods - test_start:
                    W = W.append(w, ignore_index=True).fillna(0)
                    sigma_factors = sigma_factors.append(sigma_factor, ignore_index=True).fillna(0)
                else:
                    break

        R_test = self.Data.iloc[test_start:, :].fillna(0)
        r = (R_test.iloc[self.lag:].values * sigma_factors.iloc[:-self.lag].values * W.iloc[
                                                                                     :-self.lag].values).sum(
            axis=1)

        return r, W.to_numpy()

    def parallelGridSearch(self):

        hyperparameters = self.hyperParameterGrid
        pool = Pool(os.cpu_count()-1)
        res = list(tqdm.tqdm(pool.imap(self.evaluate, hyperparameters), total=len(hyperparameters)))
        pool.close()
        pool.join()
        ret_mat = [i[0] for i in res]

        meanLeverage, meanWeight  = [], []
        for i in res:
            meanWeight.append( i[1].sum(axis=1).mean())
            meanLeverage.append(np.abs(i[1]).sum(axis=1).mean())

        performance_dict = dict(zip(hyperparameters, ret_mat))  # record the performance of each combination of hyper-parameter
        df = pd.DataFrame.from_dict(performance_dict, orient="index")

        return df, meanWeight, meanLeverage


    def summary(self, output_path):
        #global output_path
        performance_df, meanW, meanL = self.parallelGridSearch() # key is combination of hyper-parameter, value is the return vector
        result_df = pd.DataFrame(index=performance_df.index)
        result_df["annual return"] = performance_df.apply(lambda x: float(self.P * np.mean(x)), axis=1)
        result_df["annual risk"] = performance_df.apply(lambda x: float(np.sqrt(self.P) * np.std(x, ddof=0)), axis=1)
        result_df["sharpe ratio"] = result_df["annual return"] / result_df["annual risk"]
        result_df['max_drop_down'] = performance_df.apply(PeakToTrough, axis=1)
        result_df['max_time_under_water '] = performance_df.apply(UnderWater, axis=1)


        result_df["avg_leverage"] = meanL
        result_df["avg_weight"] = meanW

        result_df.round(2).to_csv(join(output_path, 'directStockSelection.csv'))

        r_df = performance_df.T
        r_df = r_df + 1
        r_df = r_df.apply(np.cumprod, axis=0)
        plot = r_df.plot(legend=True)
        fig = plot.get_figure()
        fig.savefig(join(output_path, 'Plot\\res.png'))
        plt.show()

