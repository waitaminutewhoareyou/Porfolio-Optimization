import sys
import os
from os.path import dirname, join
project_root = dirname(dirname(__file__))
solver_path = join(project_root, 'Solvers')
output_path = join(project_root, 'Result',"")
sys.path.insert(1, solver_path)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from tqdm import tqdm
from PeakToTroughAndUnderWater import PeakToTrough, UnderWater
from MarkowitzSolver import markowitzSolver
from matplotlib import pyplot as plt
from multiprocessing import Pool
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from itertools import product


class Markowitz:

    def __init__(self, Data, paramdict):
        self.Data = Data
        self.asset_names = self.Data.columns  # remember to drop the column ['DATE']
        self.num_assets = len(self.Data.columns)
        self.P = 250  # monthly frequency
        self.lag = 1
        self.gridDict = paramdict

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
        # Select all investable assets
        porfolio_universe = self.getPorfolioUniverse(self.Data.iloc[min(test_period + self.lag, num_periods - 1), :])
        R_train = self.Data[porfolio_universe]
        R_train = R_train.iloc[max(0, test_period - look_back):test_period, :]
        # Total number of NaN entries in a column must be less than 50% of total entries
        R_train = R_train.loc[:, R_train.isnull().sum() < 0.5 * R_train.shape[0]].fillna(0)

        R_train_normalized, normalization_factors = self.normalizeFactors(R_train)
        tradeable_universe = R_train_normalized.columns

        return porfolio_universe, R_train_normalized, normalization_factors

    def solve(self, solver, *args):
        return solver(*args)

    @ property
    def hyperParameterGrid(self):
        grid = self.gridDict
        lb_kappa, ub_kappa = grid['kappa']
        grid['kappa'] = np.linspace(lb_kappa, ub_kappa, num=10)

        lb_Rho, ub_Rho = grid['Rho']
        grid['Rho'] = np.linspace(lb_Rho, ub_Rho, num=10)

        lb_look_back, ub_look_back = grid['look_back']
        grid['look_back'] = np.linspace(lb_look_back, ub_look_back, num=10)

        return list(product(*grid.values()))

    def train(self, grid_point):

        Rho, kappa, look_back, freq = grid_point
        num_periods = self.Data.shape[0]
        W = pd.DataFrame(columns=self.asset_names)  # container of w's
        sigma_factors = pd.DataFrame(columns=self.asset_names)  # container of normalization factors
        test_start = look_back  # we start validation once enough data is available

        for test_period in range(test_start, num_periods, freq):

            porfolio_universe, R_train_normalized, normalization_factors = self.preProcessing(test_period, look_back,
                                                                                              num_periods)

            w_old = np.array([]) if W.empty else W[porfolio_universe].iloc[-1].fillna(0).to_numpy()

            try:
                # silence the function while solving
                save_stdout = sys.stdout
                sys.stdout = open('trash', 'w')
                partial_w = self.solve(markowitzSolver, R_train_normalized, w_old, Rho/self.P, kappa)
                sys.stdout = save_stdout

            except PermissionError:
                    partial_w = self.solve(markowitzSolver, R_train_normalized, w_old, Rho / self.P, kappa)

            # update progress
            pbar.set_postfix(
                rolling_optimization=f"{(test_period - test_start) // freq}/{(num_periods - test_start) // freq}",
                refresh=True)

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

        # df = self.quickStat(r, W).to_csv(join(output_path, f'\\Intermediate Result\\{ Rho, kappa, look_back, freq}.csv'))

        return r, W.to_numpy()

    def parallelGridSearch(self):

        hyperparameters = self.hyperParameterGrid
        pool = Pool(os.cpu_count()-1)
        res = list(tqdm(pool.imap(self.train, hyperparameters), total=len(hyperparameters)))
        pool.close()
        pool.join()
        ret_mat = [i[0] for i in res]

        meanLeverage, meanWeight  = [], []
        for i in res:
            meanWeight.append(i[1].sum(axis=1).mean())
            meanLeverage.append(np.abs(i[1]).sum(axis=1).mean())

        # record the performance of each combination of hyper-parameter
        performance_dict = dict(zip(hyperparameters, ret_mat))
        df = pd.DataFrame.from_dict(performance_dict, orient="index")

        return df, meanWeight, meanLeverage

    def quickStat(self, r, W):
        quick_df = pd.DataFrame()
        quick_df["annual return"] = float(self.P * np.mean(r))
        quick_df["annual risk"] = float(np.sqrt(self.P) * np.std(r, ddof=0))
        quick_df['sharpe ratio'] = quick_df['annual return'] / quick_df["annual risk"]
        quick_df['max_drop_down'] = PeakToTrough(r)
        quick_df['max_time_under_water'] = UnderWater(r)
        quick_df["avg_leverage"] = W.abs().sum(axis=1).mean(axis=0)
        quick_df['avg_weight'] = W.sum(axis=1).mean(axis=0)
        return quick_df

    def evaluateSharpe(self, grid_points_dict):

        grid_point = (grid_points_dict['Rho'], grid_points_dict['kappa'], int(grid_points_dict['look_back']),
                      grid_points_dict['rebalancing_frequency'])
        r, W = self.train(grid_point)
        annualized_ret = float(self.P * np.mean(r))
        annualized_std = float(np.sqrt(self.P) * np.std(r, ddof=0))

        try:
            sharpe_ratio = annualized_ret/annualized_std
        except:
            sharpe_ratio = np.nan

        return sharpe_ratio, annualized_ret

    def f(self, params):
        sharpe_ratio, ret = self.evaluateSharpe(params)

        sharpe_ls.append(sharpe_ratio)
        pbar.set_postfix(
            best_sharpe=f"{max(sharpe_ls):.2f}",
            current_sharpe= f'{sharpe_ratio:.2f}',
            refresh=True)
        pbar.update()

        return {'loss': - sharpe_ratio, 'status': STATUS_OK, "ret":ret}

    def BayesianHyperOpt(self, max_iter=100):
        global pbar
        global sharpe_ls

        pbar = tqdm(total=max_iter, desc="Hyperopt")
        sharpe_ls = []

        trials = Trials()
        lb_kappa, ub_kappa = self.gridDict['kappa']
        lb_Rho, ub_Rho = self.gridDict['Rho']
        lb_look_back, ub_look_back = self.gridDict['look_back']
        space4mark = {
            'Rho': hp.uniform('Rho', lb_Rho, ub_Rho),
            'kappa': hp.uniform('kappa',  lb_kappa, ub_kappa),
            'look_back': hp.quniform('look_back', lb_look_back,ub_look_back,1),
            'rebalancing_frequency': hp.choice('rebalancing_frequency', self.gridDict['rebalancing_frequency'])
        }

        best = fmin(self.f, space4mark, algo=tpe.suggest, max_evals=100, timeout=19 * 60 * 60, trials=trials,
                    show_progressbar=True)
        pbar.close()

        print("best:")
        print(best)

        for trial in trials.trials:
            print(trial)

        tpe_results = pd.DataFrame({'loss': [x['loss'] for x in trials.results],
                                    'annual return': [x['ret'] for x in trials.results],
                                    'iteration': trials.idxs_vals[0]['Rho'],
                                    'Rho': trials.idxs_vals[1]['Rho'],
                                    'kappa': trials.vals['kappa'],
                                    'look_back': trials.vals['look_back'],
                                    'rebalancing_frequency': [self.gridDict["rebalancing_frequency"][ix] for ix in trials.vals['rebalancing_frequency']]
                                    })

        try:
            tpe_results.to_csv(join(output_path, 'search path.csv'))
        except PermissionError:
            print(tpe_results)

        return best

    def metricsSummary(self, output_path):
        # key is combination of hyper-parameter, value is the return vector
        performance_df, meanW, meanL = self.parallelGridSearch()
        result_df = pd.DataFrame(index=performance_df.index)
        result_df["annual return"] = performance_df.apply(lambda x: float(self.P * np.mean(x)), axis=1)
        result_df["annual risk"] = performance_df.apply(lambda x: float(np.sqrt(self.P) * np.std(x, ddof=0)), axis=1)
        result_df["sharpe ratio"] = result_df["annual return"] / result_df["annual risk"]
        result_df['max_drop_down'] = performance_df.apply(PeakToTrough, axis=1)
        result_df['max_time_under_water'] = performance_df.apply(UnderWater, axis=1)
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
