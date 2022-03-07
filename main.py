# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import gpytorch
import os
import GPy
from bayes_opt import BayesianOptimization
import torch
import GPyOpt
# from GPyOpt.methods import BayesianOptimization
import numpy as np
# import plotly
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def plot_fit(x,y,mu,var, m_y='k-o', m_mu='b-<', l_y='true', l_mu='predicted', legend=True, title=''):
    """
    Plot the fit of a GP
    """
    if y is not None:
        plt.plot(x,y, m_y, label=l_y)
    plt.plot(x,mu, m_mu, label=l_mu)
    vv = 2*np.sqrt(var)
    plt.fill_between(x[:,0], (mu-vv)[:,0], (mu+vv)[:,0], alpha=0.2, edgecolor='gray', facecolor='cyan')
    if legend:
        plt.legend()
    if title != '':
        plt.title(title)




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def bay():
    pbounds = {'ol': (1.1, 3), 'al': (.5, 1.5), "bf": (.04, .18), "temp": (63, 87)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
def bbox(ol, al, bf, temp):
    data = np.array([ol,al,bf,temp]).reshape(-1, 4)
    # print(data.shape, type(data))
    predict = m.predict(Xnew=data)[0] * -1

    return predict[0][0]

def garbo():
    data = pd.read_excel("DOE_Complete.xlsx", sheet_name=1)
    print(data.info())
    data['thc'] = data['thc'].str.rstrip('%').astype("float")/100
    data['cbd'] = data['cbd'].str.rstrip('%').astype("float")/100
    data['abn_cbd'] = data['abn_cbd'].str.rstrip('%').astype("float")/100
    data['bis'] = data['bis'].str.rstrip('%').astype("float")/100
    data["fit"] = data.thc * 100 + (1 - data.cbd) + data.abn_cbd + data.bis
    print(data.head())
    # corr = data.corr()
    # sb.heatmap(corr)
    # plt.show()
    x = data.iloc[:,:4].copy().values
    print(type(x), x.shape, x[0], type(x[0]), x[0].shape)
    y = data["fit"].copy().values.reshape(-1,1)
    xt = torch.tensor(x)
    yt = torch.tensor(y)
    # initialize likelihood and model
    # print(y, y.shape)
    kernel = GPy.kern.RBF(input_dim=4, variance=1., lengthscale=1.) + GPy.kern.White(4)
    global m
    m = GPy.models.GPRegression(x, y, kernel)
    print(m)
    fig = m.plot(fixed_inputs=[(0, 1.1), (1, 0.57)])
    # GPy.plotting.show(fig, filename='basic_gp_regression_notebook')
    plt.show()
    m.optimize(messages=True)
    m.optimize_restarts(num_restarts=10)
    # GPy.plotting.show(fig, filename='basic_gp_regression_notebook_optimized')
    # fig = m.plot(plot_density=True)
    # GPy.plotting.show(fig, filename='basic_gp_regression_density_notebook_optimized')
    print(m)
    # print(m.predict(Xnew=np.array([[0.917022004702574, 0.14084542908190212, 1.1002173121529553, 70.25598174316416]])))
    # print(dir(m))
    pbounds = {'ol': (1.1, 3), 'al': (.5, 1.5), "bf": (.04, .18), "temp": (63, 87)}
    # print(bbox(0.917022004702574, 0.14084542908190212, 1.1002173121529553, 70.25598174316416))
    optimizer = BayesianOptimization(
        f=bbox,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1
    )
    optimizer.maximize(init_points=5, n_iter = 5)
    print(optimizer.max)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # GPy.plotting.change_plotting_library('plotly')
    print_hi('PyCharm')
    garbo()
    # bay()
