import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import metrics
import matplotlib.gridspec as gridspec

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Importing package for Latin Hypercube Sampling (LHS).
from smt.sampling_methods import LHS

# Importing package for train/test split.
from sklearn.model_selection import train_test_split

# Importing regression models
# Importing LinearRegression for the Multiple Linear Regression.
from sklearn.linear_model import LinearRegression

# Importing GaussionProcessRegressor and the kernels (RBF, WhiteKernel)
# for the eponymous regression.
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Importing SVM and SVR for the Support Vector Regression.
from sklearn import svm
from sklearn.svm import SVR

# Importing packages for kFold crossvalidation.
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score


class surrogate_modelling_wingweight:
    """
    Surrogate modelling of the wing weight function class consist of various
    properties.

    ...
    
    Attributes
    ----------
    dataset_size: int
        size of the generated dataset on which the model is trained,
        tested and the outcomes are being predicted
    '''
    
    Methods
    ----------
        dataset_LHS(self, dataset_size):
            generates the dataset with the given dataset size,
            using the latin hypercube sampling method
        wingweight_function(self, diameter, length):
            calculates the weight of the wing with the 10 given variables
        ml_regression(self, regression_type):
            predicts the outcomes of the surrogate model using multiple
            linear regression
        gp_regression(self, regression_type):
            predicts the outcomes of the surrogate model using gaussina
            process regression
        sv_regression(self, regression_type):
            predicts the outcomes of the surrogate model using support
            vector regression
        model_evaluation(self, ...):
            measures the model performance and returns the best suited
            approach to the problem

    """

    def __init__(self, dataset_size):
        """ Constructs all the necessary attributes for the material object.

        Parameters
        ----------
            dataset_size: int
                size of the generated dataset on which the model is trained,
                tested and the outputs are being predicted on
                
        Raises
        ----------
            Exception: if file index is not in the range of files in
            the article
            TypeError: if rows_to_skip is not an integer
            Exception: if files in the article are not csv or xlsx
        """
        self.dataset_size = dataset_size
        self.dataset_LHS(dataset_size)
        self.train_test_split()
        self.ml_regression()
        self.gp_regression()
        self.sv_regression()
        self.regression_plot()
        self.panel_plot()
        self.model_evaluation()

        if not isinstance(dataset_size, int):
            raise TypeError("dataset_size must be an integer")

    def dataset_LHS(self, dataset_size):
        """ Generates the dataset by applying the LHS method.

        Parameters
        ----------
            dataset_size: int
                size of the generated dataset on which the model is trained,
                tested and the outputs are being predicted on
                
        Raises
        ----------
            -
        
        Returns
        ----------
            data_wingweight : Pandas DataFrame
                Pandas dataframe of the generated dataset
        """
        if not isinstance(dataset_size, int):
            raise TypeError("dataset_size must be an integer")
        if dataset_size < 31:
            raise Exception("dataset_size must be greater than 30")

        # Dictionary with all parameters an their value ranges.
        input_domain = {
            "S_w": [150, 200],
            "W_fw": [220, 300],
            "A": [6, 10],
            "LamCaps": [-10, 10],
            "q": [16, 45],
            "lam": [0.5, 1],
            "t_c": [0.08, 0.18],
            "N_z": [2.5, 6],
            "W_dg": [1700, 2500],
            "W_p": [0.025, 0.08],
        }

        xlimits = np.array(
            [
                input_domain["S_w"],
                input_domain["W_fw"],
                input_domain["A"],
                input_domain["LamCaps"],
                input_domain["q"],
                input_domain["lam"],
                input_domain["t_c"],
                input_domain["N_z"],
                input_domain["W_dg"],
                input_domain["W_p"],
            ]
        )

        sampling = LHS(xlimits=xlimits, random_state=1)

        x = sampling(dataset_size)

        data_wingweight = pd.DataFrame(
            x,
            columns=[
                "S_w",
                "W_fw",
                "A",
                "LamCaps",
                "q",
                "lam",
                "t_c",
                "N_z",
                "W_dg",
                "W_p",
            ],
        )

        def wing_weight(data_wingweight):
            """ Calculates the wing weight for the given input variables.

            Parameters
            ----------
                data_wingweight : Pandas DataFrame
                    Pandas dataframe of the generated dataset

            Raises
            ----------
                -

            Returns
            ----------
                y_wingweight : Pandas DataFrame
                    Pandas dataframe of the generated y_wingweight dataset
            """
            fact1 = (
                0.036
                * data_wingweight["S_w"] ** 0.758
                * data_wingweight["W_fw"] ** 0.0035
            )
            fact2 = (
                data_wingweight["A"]
                / ((math.cos(data_wingweight["LamCaps"] * (math.pi / 180))) ** 2)
            ) ** 0.6
            fact3 = data_wingweight["q"] ** 0.006 * data_wingweight["lam"] ** 0.04
            fact4 = (
                100
                * data_wingweight["t_c"]
                / math.cos(data_wingweight["LamCaps"] * (math.pi / 180))
            ) ** (-0.3)
            fact5 = (data_wingweight["N_z"] * data_wingweight["W_dg"]) ** 0.49

            term1 = data_wingweight["S_w"] * data_wingweight["W_p"]

            y_wingweight = fact1 * fact2 * fact3 * fact4 * fact5 + term1

            return y_wingweight

        # Extending the data_wingweight dataset with y_wingweight
        data_wingweight["y_wingweight"] = data_wingweight.apply(wing_weight, axis=1)
        self.data_wingweight = data_wingweight

        # Returning the full dataset with the 10 variables and the dependend 11th
        return self.data_wingweight

    def train_test_split(self):
        """ Splits the generated dataset data_wingweight into a train and test subset.

        Parameters
        ----------
        -
                
        Returns
        ----------
            X: Numpy Array
                Numpy array of the 10 variables
            y: Numpy Array
                Numpy array of the y_wingweight values
            X_train: Numpy Array
                Numpy array of the training data of the X dataframe
            X_test: Numpy Array
                Numpy array of the test data of the X dataframe
            y_train: Numpy Array
                Numpy array of the training data of the y dataframe
            y_test: Numpy Array
                Numpy array of the test data of the y dataframe
        """
        # Split the data_wingweight_full dataset into input variables and output.
        self.X = self.data_wingweight.iloc[:, 0:10].values
        self.y = self.data_wingweight.iloc[:, 10].values

        # Splitting the data_wingweight dataset into train and test sets with
        # a 70/30 split.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=100
        )

        return (self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test)

    def ml_regression(self):
        """ Predicting the output value of the wing weight function with the MLR.

        Parameters
        ----------
        -
                
        Returns
        ----------
            y_pred_mlr: Numpy Array
                Numpy array of predicted values of the X_test data
            y_pred_full_mlr: Numpy Array
                Numpy array of predicted values of the X data
        """
        # Fitting the MLR model.
        self.mlr = LinearRegression()
        self.mlr.fit(self.X_train, self.y_train)

        # Prediction of test set.
        self.y_pred_mlr = self.mlr.predict(self.X_test)

        # Prediction of full dataset.
        self.y_pred_full_mlr = self.mlr.predict(self.X)

        return (self.y_pred_mlr, self.y_pred_full_mlr)

    def gp_regression(self):
        """ Predicting the output value of the wing weight function with the GPR.

        Parameters
        ----------
        -
                
        Returns
        ----------
            y_pred_gpr: Numpy Array
                Numpy array of predicted values of the X_test data
            y_pred_full_gpr: Numpy Array
                Numpy array of predicted values of the X data
        """
        # Fitting the GPR model.
        noise_std = 0.05
        kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        self.model = GaussianProcessRegressor(
            kernel=kernel, alpha=noise_std ** 2, n_restarts_optimizer=9
        )

        self.model.fit(self.X_train, self.y_train)

        # Prediction of test set.
        self.y_pred_gpr = self.model.predict(self.X_test)

        # Prediction of full dataset.
        self.y_pred_full_gpr, self.std_prediction = self.model.predict(
            self.X, return_std=True
        )

        return (self.y_pred_full_gpr, self.y_pred_gpr)

    def sv_regression(self):
        """ Predicting the output value of the wing weight function with the SVR.

        Parameters
        ----------
        -
                
        Returns
        ----------
            y_pred_svr: Numpy Array
                Numpy array of predicted values of the X_test data
            y_pred_full_svr: Numpy Array
                Numpy array of predicted values of the X data
        """
        # Fitting the GPR model.
        self.svr_sigmoid = SVR(kernel="sigmoid", C=1e3, degree=2)
        self.svr_sigmoid.fit(self.X_train, self.y_train)

        # Prediction of test set.
        self.y_pred_svr = self.svr_sigmoid.predict(self.X_test)

        # Prediction of full dataset.
        self.y_pred_full_svr = self.svr_sigmoid.predict(self.X)

        return (self.y_pred_svr, self.y_pred_full_svr)

    def regression_plot(self):
        """ Plots the predicted and the observed output of the regression models.

        Parameters
        ----------
        -
                
        Returns
        ----------
        -
        """
        gs = gridspec.GridSpec(1, 3)
        plt.figure(figsize=(30, 8), dpi=80)
        plt.rcParams["font.size"] = "20"

        ax = plt.subplot(gs[0, 0])
        plt.scatter(self.y, self.y_pred_full_mlr)
        plt.plot(
            [100, self.data_wingweight["y_wingweight"].max()],
            [100, self.data_wingweight["y_wingweight"].max()],
            linestyle="--",
            color="k",
            linewidth=2,
            label="Reference line (predicted = observed)",
        )
        plt.xlabel("Actual output")
        plt.ylabel("Predicted output")
        plt.title("MLR")

        ax = plt.subplot(gs[0, 1])
        plt.scatter(self.y_test, self.y_pred_gpr)
        plt.plot(
            [100, self.data_wingweight["y_wingweight"].max()],
            [100, self.data_wingweight["y_wingweight"].max()],
            linestyle="--",
            color="k",
            linewidth=2,
            label="Reference line (predicted = observed)",
        )
        plt.xlabel("Actual output")
        plt.title("GPR")

        ax = plt.subplot(gs[0, 2])
        plt.scatter(self.y_test, self.y_pred_svr, label="predicted vs. actual")
        plt.plot(
            [100, self.data_wingweight["y_wingweight"].max()],
            [100, self.data_wingweight["y_wingweight"].max()],
            linestyle="--",
            color="k",
            linewidth=2,
            label="Reference line (predicted = observed)",
        )
        plt.xlabel("Actual output")
        plt.title("SVR")

        plt.suptitle("Comparison of predicted and actual output")
        plt.legend()

    def panel_plot(self):
        """ Plots the predicted values for the wingweight and easch variable
            in a panel plot.

        Parameters
        ----------
        -
                
        Returns
        ----------
        -
        """
        list_regression = [
            self.y_pred_full_mlr,
            self.y_pred_full_gpr,
            self.y_pred_full_svr,
        ]
        list_color = ["red", "green", "blue"]
        list_color_dots = ["lightcoral", "lightgreen", "lightskyblue"]
        list_titles = ["MLR", "GPR", "SVR"]

        for j in range(0, len(list_regression)):

            rows = 2
            columns = 5

            plt.figure(
                figsize=(25 / 2.54, 15 / 2.54), dpi=600, facecolor="w", edgecolor="w"
            )
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = "10"
            grid = plt.GridSpec(rows, columns, wspace=0.07, hspace=0.4)

            wingweight_names = [
                "S_w",
                "W_fw",
                "A",
                "LamCaps",
                "q",
                "lam",
                "t_c",
                "N_z",
                "W_dg",
                "W_p",
            ]

            for i in range(0, 10):
                # Rearranging the data for the panel plot.
                plt.subplot(grid[i])
                plt.scatter(
                    self.data_wingweight.iloc[:, i],
                    list_regression[j],
                    color=list_color_dots[j],
                    s=1,
                )
                plt.ylim(100, list_regression[j].max())
                plt.xlim(
                    self.data_wingweight.iloc[:, i].min(),
                    self.data_wingweight.iloc[:, i].max(),
                )

                plt.title(wingweight_names[i])

                # Plotting the regression line.
                m, b = np.polyfit(
                    self.data_wingweight.iloc[:, i], list_regression[j], 1
                )
                plt.plot(
                    self.data_wingweight.iloc[:, i],
                    m * self.data_wingweight.iloc[:, i] + b,
                    color=list_color[j],
                )

                ax = plt.gca()
                plt.yticks(np.arange(100, list_regression[j].max(), 10))

                # Xticks for S_w and W_fw
                if (
                    self.data_wingweight.iloc[:, i].min() > 140
                    and self.data_wingweight.iloc[:, i].max() < 350
                ):
                    plt.xticks(
                        np.arange(
                            self.data_wingweight.iloc[:, i].min(),
                            self.data_wingweight.iloc[:, i].max(),
                            10,
                        ),
                        rotation=45,
                    )
                # Xticks for LamCaps
                elif (
                    self.data_wingweight.iloc[:, i].min() < -9
                    and self.data_wingweight.iloc[:, i].max() < 11
                ):
                    plt.xticks(
                        np.arange(
                            self.data_wingweight.iloc[:, i].min(),
                            self.data_wingweight.iloc[:, i].max(),
                            2,
                        ),
                        rotation=45,
                    )
                # Xticks for A
                elif (
                    self.data_wingweight.iloc[:, i].min() > 5
                    and self.data_wingweight.iloc[:, i].max() < 11
                ):
                    plt.xticks(
                        np.arange(
                            self.data_wingweight.iloc[:, i].min(),
                            self.data_wingweight.iloc[:, i].max(),
                            1,
                        ),
                        rotation=45,
                    )
                # Xticks for N_z
                elif (
                    self.data_wingweight.iloc[:, i].min() > 2
                    and self.data_wingweight.iloc[:, i].max() < 7
                ):
                    plt.xticks(
                        np.arange(
                            self.data_wingweight.iloc[:, i].min(),
                            self.data_wingweight.iloc[:, i].max(),
                            1,
                        ),
                        rotation=45,
                    )
                # Xticks for q
                elif (
                    self.data_wingweight.iloc[:, i].min() > 15
                    and self.data_wingweight.iloc[:, i].max() < 50
                ):
                    plt.xticks(
                        np.arange(
                            self.data_wingweight.iloc[:, i].min(),
                            self.data_wingweight.iloc[:, i].max(),
                            5,
                        ),
                        rotation=45,
                    )
                # Xticks for t_c
                elif self.data_wingweight.iloc[:, i].max() < 0.20:
                    plt.xticks(
                        np.arange(
                            self.data_wingweight.iloc[:, i].min(),
                            self.data_wingweight.iloc[:, i].max(),
                            0.02,
                        ),
                        rotation=45,
                    )
                # Xticks for lam
                elif (
                    self.data_wingweight.iloc[:, i].min() > 0.4
                    and self.data_wingweight.iloc[:, i].max() < 1.1
                ):
                    plt.xticks(
                        np.arange(
                            self.data_wingweight.iloc[:, i].min(),
                            self.data_wingweight.iloc[:, i].max(),
                            0.1,
                        ),
                        rotation=45,
                    )
                # Xticks for W_dg
                elif (
                    self.data_wingweight.iloc[:, i].min() > 1650
                    and self.data_wingweight.iloc[:, i].max() < 2600
                ):
                    plt.xticks(
                        np.arange(
                            self.data_wingweight.iloc[:, i].min(),
                            self.data_wingweight.iloc[:, i].max(),
                            100,
                        ),
                        rotation=45,
                    )
                else:
                    plt.xticks(
                        np.arange(
                            self.data_wingweight.iloc[:, i].min(),
                            self.data_wingweight.iloc[:, i].max(),
                            0.02,
                        ),
                        rotation=45,
                    )
                ax.axes.get_yaxis().set_ticklabels([])

            for i in range(rows):
                exec(f"plt.subplot(grid[{i}, 0])")
                plt.ylabel("Wing weight [lb]")
                plt.yticks(
                    np.arange(100, list_regression[j].max(), 50),
                    labels=np.arange(100, list_regression[j].max(), 50),
                )

            plt.suptitle(list_titles[j])

    def model_evaluation(self):
        """ Plots the predicted the kFold corssvalidation regession model performance
            in a boxplot.

        Parameters
        ----------
        -
                
        Returns
        ----------
        -
        """

        models = []

        models.append(("MLR", self.mlr))
        models.append(("GPR", self.model))
        models.append(("SVR", self.svr_sigmoid))

        gs = gridspec.GridSpec(1, 3)
        fig = plt.figure(figsize=(30, 8), dpi=80)
        plt.rcParams["font.size"] = "20"

        ax = plt.subplot(gs[0, 0])
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(
                model, self.X_test, self.y_test, scoring="r2", cv=cv, n_jobs=-1
            )
            results.append(absolute(n_scores))
            names.append(name)
            msg = "%s: %f (%f)" % (
                name,
                mean(absolute(n_scores)),
                std(absolute(n_scores)),
            )
            print("R2:", msg)
        # boxplot algorithm comparison
        plt.title("R-Squared (R2)")
        plt.boxplot(results)
        plt.ylim([0, 1])
        ax.set_xticklabels(names)
        plt.ylabel("R2 score")

        ax = plt.subplot(gs[0, 1])
        # evaluate each model in turn
        results1 = []
        names1 = []
        for name, model in models:
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(
                model,
                self.X_test,
                self.y_test,
                scoring="neg_mean_squared_error",
                cv=cv,
                n_jobs=-1,
            )
            results1.append(absolute(n_scores))
            names1.append(name)
            msg1 = "%s: %f (%f)" % (
                name,
                mean(absolute(n_scores)),
                std(absolute(n_scores)),
            )
            print("MSE:", msg1)
        # boxplot algorithm comparison
        plt.title("Mean Squared Error (MSE)")
        plt.boxplot(results1)
        ax.set_xticklabels(names1)
        plt.ylabel("MSE score")

        ax = plt.subplot(gs[0, 2])
        # evaluate each model in turn
        results2 = []
        names2 = []
        for name, model in models:
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(
                model,
                self.X_test,
                self.y_test,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
            )
            results2.append(absolute(n_scores))
            names2.append(name)
            msg2 = "%s: %f (%f)" % (
                name,
                mean(absolute(n_scores)),
                std(absolute(n_scores)),
            )
            print("RMSE:", msg2)
        # boxplot algorithm comparison
        plt.title("Root Mean Squared Error (RMSE)")
        plt.boxplot(results2)
        ax.set_xticklabels(names2)
        plt.ylabel("RMSE score")

        plt.suptitle("Regression Model Validation")
