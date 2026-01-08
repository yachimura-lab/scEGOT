import copy
import itertools
import warnings
from collections import defaultdict
from io import BytesIO

import anndata
import cellmap
import matplotlib.animation as animation
import matplotlib.collections as collect
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydotplus
import scipy.linalg as spl
import scipy.sparse.linalg as spl_sparse
import screcode
import scvelo as scv
import seaborn as sns
import umap.umap_ as umap
from adjustText import adjust_text
from IPython.display import HTML, Image, display
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from PIL import Image as PILImage
from scanpy.pp import neighbors
from scipy import interpolate
from scipy.sparse import csc_matrix, issparse, lil_matrix
from scipy.stats import multivariate_normal, zscore
from sklearn import linear_model
from sklearn.base import clone as sklearn_clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import check_random_state
from tqdm import tqdm

sns.set_style("whitegrid")


def is_notebook():
    """
    Check if the code is running in a Jupyter notebook or not.

    Returns
    -------
    bool
        True if the code is running in a Jupyter notebook, False otherwise.
    """

    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook or qtconsole
            return True
        elif shell == "TerminalInteractiveShell":
            # Terminal running IPython
            return False
        else:
            # Other type
            return False
    except NameError:
        # Probably standard Python interpreter
        return False


def _check_input_data(input_data, day_names, adata_day_key):
    """
    Check the input data and return the processed data.

    Parameters
    ----------
    input_data : any
        Input data.
        
    day_names : list of str
        List of day names.
        
    adata_day_key : str
        AnnData observation key for day names.

    Raises
    ------
    ValueError
        When 'X' is an array of DataFrame and 'day_names' is not specified
        or 'X' is AnnData and 'adata_day_key' is not specified.
        
    TypeError
        When 'X' is not an array of DataFrame or AnnData.

    Returns
    -------
    list of pd.DataFrame
        List of DataFrames.
    """

    if isinstance(input_data, list):
        if day_names is None:
            raise ValueError(
                "When 'X' is an array of DataFrame, 'day_names' should be specified."
            )
        for df in input_data:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("'X' should be an array of DataFrames.")
        return input_data, day_names

    elif isinstance(input_data, anndata.AnnData):
        if adata_day_key is None:
            raise ValueError(
                "When 'X' is AnnData, 'adata_day_key' should be specified."
            )

        print("Processing AnnData...")

        if issparse(input_data.X):
            X_concated = pd.DataFrame.sparse.from_spmatrix(
                input_data.X, index=input_data.obs.index, columns=input_data.var.index
            )
        else:
            X_concated = pd.DataFrame(
                input_data.X, index=input_data.obs.index, columns=input_data.var.index
            )

        if day_names is None:
            day_names = pd.Series(input_data.obs[adata_day_key]).unique().tolist()

        X = []
        for c_ in day_names:
            X.append(X_concated[input_data.obs[adata_day_key] == c_])

        return X, day_names

    else:
        raise TypeError("'X' should be AnnData or an array of DataFrames.")


def integrate_data(
    input_data_dict,
    adata_day_key=None,
    recode_params={},
    recode_fit_transform_params={},    
):
    """
    Integrate multiple data using iRECODE.

    Parameters
    ----------
    input_data_dict : dict
        A dictionary where keys are data names and values are
        input data (list of pd.DataFrame or AnnData).  

    adata_day_key : str, optional
        Name of the key in AnnData.obs for day names, by default None.
        Should be specified when values of input_data_dict are AnnData.

    recode_params : dict, optional
        paramaters passed to the screcode.RECODE constructor, by default {}

    recode_fit_transform_params : dict, optional
        paramaters for RECODE.fit_transform(), by default {}

    Raises
    ------
    ValueError 
        When 'X' is AnnData and 'adata_day_key' is not specified.
    
    TypeError
        This error is raised in the following cases:
        
        * When input_data_dict is not a dict.
        * When values of input_data_dict is neither list of pd.DataFrame nor AnnData.  
        
    Returns
    -------
    (list of pd.DataFrame) or AnnData
        Integrated data.
    """

    input_type_error_msg = (
        "input_data_dict must be a dict with values as a list of AnnData or a list of list of DataFrame."
    )

    if not isinstance(input_data_dict, dict):
        raise TypeError(input_type_error_msg)

    input_data_list = input_data_dict.values()

    if all(isinstance(data, anndata.AnnData) for data in input_data_list):
        if adata_day_key is None:
            raise ValueError("When 'X' is AnnData, 'adata_day_key' should be specified.")
        for data_name, adata in input_data_dict.items():
            adata.obs.index = data_name + "_" + adata.obs.index
            adata.obs["batch"] = data_name + "_" + adata.obs[adata_day_key].astype(str)
        concated_adata = anndata.concat(input_data_list)
        recode = screcode.RECODE(**recode_params)
        integrated_adata = recode.fit_transform(
            concated_adata,
            batch_key="batch",
            **recode_fit_transform_params
        )
        integrated_adata.X = integrated_adata.layers["RECODE"]
        return integrated_adata

    elif all(isinstance(data, list) for data in input_data_list):
        metadata_df_list = []
        for data_name, df_list in input_data_dict.items():
            if not all(isinstance(df, pd.DataFrame) for df in df_list):
                raise TypeError(input_type_error_msg)
            for day_index, day_df in enumerate(df_list):
                day_df.rename(index= lambda s: f"{data_name}_{s}", inplace=True)
                n_day_data = len(day_df)
                metadata_df = pd.DataFrame(
                    [f"{data_name}_{day_index}"]*n_day_data,
                    columns=["batch"],
                    index=day_df.index
                )
                metadata_df_list.append(metadata_df)
        concated_df = pd.concat(list(itertools.chain.from_iterable(input_data_list)))
        concated_metadata_df = pd.concat(metadata_df_list)

        recode = screcode.RECODE(**recode_params)
        integrated_array = recode.fit_transform(
            concated_df.values,
            meta_data=concated_metadata_df,
            batch_key="batch",
            **recode_fit_transform_params
        )
        integrated_df = pd.DataFrame(
            integrated_array,
            columns=concated_df.columns,
            index=concated_df.index
        )

        new_df_list = []
        day_num = max([len(df_list) for df_list in input_data_list])
        for i in range(day_num):
            day_mask = concated_metadata_df["batch"].str.endswith(f"_{i}")
            new_df_list.append(integrated_df[day_mask])
        return new_df_list

    else:
        raise TypeError(input_type_error_msg)


class scEGOT:
    def __init__(
        self,
        X,
        day_names=None,
        verbose=True,
        adata_day_key=None,
    ):
        """
        Initialize the scEGOT object.

        Parameters
        ----------
        X : list of pd.DataFrame or AnnData
            Input data.
        day_names : list of str, optional
            List of day names. Defaults to None.
            Should be specified when 'X' is an array of DataFrame.
            The order of the day names should be the same as the order of the data.
        verbose : bool, optional
            If False, all running messages are not displayed. Defaults to True.
        adata_day_key : str, optional
            AnnData observation key for day names. Defaults to None.
            Should be specified when 'X' is AnnData.
            Day names are extracted from the specified key.
            The order of the day names will be the same as the order of appearance in the data.

        Attributes
        ----------
        X_raw : list of pd.DataFrame of shape (n_samples, n_genes)
            Raw input data.
            Arranged in the order of the day names.

        X_normalized : list of pd.DataFrame of shape (n_samples, n_genes) or None
            Normalized input data.
            This attribute is None before the 'preprocess' method is called.

        X_selected : list of pd.DataFrame of shape (n_samples, n_highly_variable_genes) or None
            Filtered input data with highly variable genes after normalization.
            If 'select_genes' is False in the 'preprocess' method, this will be the same as 'X_normalized'.
            This attribute is None before the 'preprocess' method is called.

        X_pca : list of pd.DataFrame of shape (n_samples, n_components of PCA) or None
            PCA-transformed input data.
            PCA is applied to the X_selected data.
            This attribute is None before the 'preprocess' method is called.
        
        X_umap : list of pd.DataFrame of shape (n_samples, n_components of UMAP) or None
            UMAP-transformed input data.
            UMAP is applied to the X_pca data.
            This attribute is None before the 'apply_umap' method is called.

        pca_model : PCA or None
            PCA model.
            This attribute is None before the 'preprocess' method is called.

        gmm_n_components_list : list of int or None
            List of the number of components for GMM.
            Each element corresponds to the number of components of the GMM model for each day.
            This attribute is None before the 'fit_gmm' or 'fit_predict_gmm' method is called.

        gmm_models : list of GaussianMixture or None
            List of GMM models.
            This attribute is None before the 'fit_gmm' or 'fit_predict_gmm' method is called.

        gmm_labels : list of np.ndarray or None
            List of GMM labels.
            This attribute is None before the 'fit_predict_gmm' method is called.
    
        gmm_labels_modified : list of np.ndarray or None
            List of modified GMM labels.
            This attribute is None before the 'fit_predict_gmm' method is called.

        gmm_label_converter : list of np.ndarray or None
            List of GMM label converters.
            This attribute is None before the 'replace_gmm_labels' method is called.

        umap_model : UMAP
            UMAP model.
            This attribute is None before the 'apply_umap' method is called.

        day_names : list of str
            List of day names.

        gene_names : pd.Index
            Gene names.
            If you call the 'preprocess' method with 'select_genes' set to True, 
            this attribute will be the highly variable gene names.
            Otherwise, this attribute will be the gene names of the input data.

        solutions : list of np.ndarray
            List of solutions.
        """

        self.verbose = verbose

        X, day_names = _check_input_data(X, day_names, adata_day_key)

        self.X_raw = [df.copy() for df in X]
        self.X_normalized = None
        self.X_selected = None
        self.X_pca = None
        self.X_umap = None

        self.pca_model = None
        self.gmm_n_components_list = None
        self.gmm_models = None
        self.gmm_labels = None
        self.gmm_labels_modified = None
        self.gmm_label_converter = None
        self.umap_model = None

        self.day_names = day_names
        self.gene_names = None

        self.solutions = None

    def _preprocess_recode(self, X_concated, recode_params={}):
        X_concated = pd.DataFrame(
            screcode.RECODE(
                verbose=self.verbose,
                **recode_params,
            ).fit_transform(X_concated.values),
            index=X_concated.index,
            columns=X_concated.columns,
        )
        return X_concated

    def _preprocess_pca(
        self, X_concated, n_components, random_state=None, pca_other_params={}
    ):
        pca_model = PCA(
            n_components=n_components,
            random_state=random_state,
            **pca_other_params,
        )
        X_concated = pd.DataFrame(
            pca_model.fit_transform(X_concated.values),
            index=X_concated.index,
            columns=["PC{}".format(i + 1) for i in range(n_components)],
        )
        return X_concated, pca_model

    def _normalize_umi(self, X_concated, target_sum=1e4):
        X_concated = X_concated.div(X_concated.sum(axis=1), axis=0) * target_sum
        return X_concated

    def _normalize_log1p(self, X_concated):
        X_concated = X_concated.where(X_concated > 0, 0)
        X_concated = pd.DataFrame(
            np.log1p(X_concated.values),
            index=X_concated.index,
            columns=X_concated.columns,
        )
        return X_concated

    def _normalize_data(self, X_concated, target_sum=1e4):
        X_concated = self._normalize_umi(X_concated, target_sum)
        X_concated = self._normalize_log1p(X_concated)
        return X_concated

    def _select_highly_variable_genes(self, X_concated, n_select_genes=2000, hvg_method="dispersion", **recode_params):
        if hvg_method not in ["dispersion", "RECODE"]:
            raise ValueError("The parameter 'hvg_method' should be 'dispersion' or 'RECODE'.")
        
        genes = pd.DataFrame(index=X_concated.columns)
        
        if hvg_method == "dispersion":
            mean = X_concated.values.mean(axis=0)
            mean[mean == 0] = 1e-12
            var_norm = X_concated.values.var(axis=0) / mean
            var_norm[var_norm == 0] = np.nan
            genes["dispersion"] = var_norm
            highvar_gene_names = (
                genes.sort_values(by=["dispersion"], ascending=False)
                .head(n_select_genes)
                .index
            )
        else:
            adata = anndata.AnnData(X_concated.values)
            adata.obs_names = X_concated.index
            adata.var_names = X_concated.columns
            recode = screcode.RECODE(verbose=self.verbose, **recode_params)
            adata = recode.fit_transform(adata)
            recode.highly_variable_genes(adata, n_top_genes=n_select_genes) 
            highvar_gene_names = adata.var.index[adata.var["RECODE_highly_variable"]]

        highvar_genes = X_concated.loc[:, highvar_gene_names]
        return highvar_genes

    def _split_dataframe_by_row(self, df, row_counts):
        split_indices = list(itertools.accumulate(row_counts))
        df_list = [
            df.iloc[(split_indices[i - 1] if i > 0 else 0) : split_indices[i]]
            for i in range(len(split_indices))
        ]
        return df_list

    def preprocess(
        self,
        pca_n_components,
        recode_params={},
        umi_target_sum=1e4,
        pca_random_state=None,
        pca_other_params={},
        apply_recode=True,
        apply_normalization_log1p=True,
        apply_normalization_umi=True,
        select_genes=True,
        n_select_genes=2000,
        hvg_method="dispersion",
    ):
        """
        Preprocess the input data. 
        
        Apply scRECODE, normalize, select highly variable genes, and apply PCA.

        Parameters
        ----------
        pca_n_components : int
            Number of components to keep in PCA.
            Passed to the 'n_components' parameter of the PCA class.
            
        recode_params : dict, optional
            Parameters for scRECODE, by default {}
            
        umi_target_sum : int or float, optional
            Target sum for UMI normalization, by default 1e4
            
        pca_random_state : int, RandomState instance or None, optional
            Pass an int for reproducible results, by default None
            Passed to the 'random_state' parameter of the PCA class.
            
        pca_other_params : dict, optional
            Parameters other than 'n_components' and 'random_state' for PCA, by default {}
            
        apply_recode : bool, optional
            If True, apply scRECODE, by default True
            
        apply_normalization_log1p : bool, optional
            If True, apply log1p normalization, by default True
            
        apply_normalization_umi : bool, optional
            If True, apply UMI normalization, by default True
            
        select_genes : bool, optional
            If True, filter genes and select highly variable genes, by default True
            
        n_select_genes : int, optional
            Number of highly variable genes to select, by default 2000
            Used only when 'select_genes' is True.
        
        hvg_method : {'dispersion', 'RECODE'}, optional
            Method to select highly variable genes, by default 'dispersion'
            * 'dispersion': select genes based on dispersion.
            * 'RECODE': select genes based on scRECODE.
        
        Raises
        ------
        ValueError
            If 'hvg_method' is not 'dispersion' or 'RECODE'.

        Returns
        -------
        list of pd.DataFrame of shape (n_samples, n_components of PCA)
            Normalized, filtered, and PCA-transformed data.
            
        sklearn.decomposition.PCA
            PCA instance fitted to the input data.
        """

        if hvg_method not in ["dispersion", "RECODE"]:
            raise ValueError("The parameter 'hvg_method' should be 'dispersion' or 'RECODE'.")
        
        X_concated = pd.concat(self.X_raw)

        if apply_recode:
            if self.verbose:
                print("Applying RECODE...")
            X_concated = self._preprocess_recode(
                X_concated,
                recode_params,
            )

        if apply_normalization_umi:
            if self.verbose:
                print("Applying UMI normalization...")
            X_concated = self._normalize_umi(X_concated, umi_target_sum)

        if apply_normalization_log1p:
            if self.verbose:
                print("Applying log1p normalization...")
            X_concated = self._normalize_log1p(X_concated)

        self.X_normalized = self._split_dataframe_by_row(
            X_concated.copy(), [len(x) for x in self.X_raw]
        )

        if select_genes:
            X_concated = self._select_highly_variable_genes(
                X_concated,
                n_select_genes=n_select_genes,
                hvg_method=hvg_method
            )

        self.gene_names = X_concated.columns

        self.X_selected = self._split_dataframe_by_row(
            X_concated.copy(), [len(x) for x in self.X_raw]
        )

        if self.verbose:
            print("Applying PCA...")
        X_concated, pca_model = self._preprocess_pca(
            X_concated, pca_n_components, pca_random_state, pca_other_params
        )

        if self.verbose:
            print(
                f"\tsum of explained_variance_ratio = {sum(pca_model.explained_variance_ratio_ * 100)}"
            )

        X = self._split_dataframe_by_row(X_concated, [len(x) for x in self.X_raw])

        self.X_pca = [df.copy() for df in X]
        self.pca_model = pca_model

        return X, pca_model

    def _apply_umap_to_concated_data(
        self,
        X_concated,
        n_neighbors,
        n_components,
        random_state=None,
        min_dist=0.1,
        umap_other_params={},
    ):
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=random_state,
            min_dist=min_dist,
            **umap_other_params,
        )
        X_concated = pd.DataFrame(
            umap_model.fit_transform(X_concated.values),
            index=X_concated.index,
            columns=["UMAP1", "UMAP2"],
        )
        return X_concated, umap_model

    def apply_umap(
        self,
        n_neighbors,
        n_components=2,
        random_state=None,
        min_dist=0.1,
        umap_other_params={},
    ):
        """
        Fit self.X_pca to UMAP and return the transformed data.

        Parameters
        ----------
        n_neighbors : float
            The size of local neighborhood used for manifold approximation.
            Passed to the 'n_neighbors' parameter of the UMAP class.
            
        n_components : int, optional
            The dimension of the space to embed into, by default 2
            Passed to the 'n_components' parameter of the UMAP class.
            
        random_state : int, RandomState instance or None, optional
            Fix the random seed for reproducibility, by default None
            Passed to the 'random_state' parameter of the UMAP class.
            
        min_dist : float, optional
            The effective minimum distance between embedded points, by default 0.1
            Passed to the 'min_dist' parameter of the UMAP class.
            
        umap_other_params : dict, optional
            Other parameters for UMAP, by default {}

        Returns
        -------
        list of pd.DataFrame of shape (n_samples, n_components of UMAP)
            UMAP-transformed data.
            
        umap.umap\_.UMAP
            UMAP instance fitted to the input data.
        """

        X_concated = pd.concat(self.X_pca)
        X_concated, umap_model = self._apply_umap_to_concated_data(
            X_concated,
            n_neighbors,
            n_components,
            random_state,
            min_dist,
            umap_other_params,
        )
        X = self._split_dataframe_by_row(X_concated, [len(x) for x in self.X_raw])

        self.X_umap = [df.copy() for df in X]
        self.umap_model = umap_model

        return X, umap_model

    def fit_gmm(
        self,
        n_components_list,
        covariance_type="full",
        max_iter=2000,
        n_init=10,
        random_state=None,
        gmm_other_params={},
    ):
        gmm_models = []

        if self.verbose:
            print("Fitting GMM models with each day's data...")
        for i in (
            tqdm(range(len(self.X_pca))) if self.verbose else range(len(self.X_pca))
        ):
            gmm_model = GaussianMixture(
                n_components_list[i],
                covariance_type=covariance_type,
                max_iter=max_iter,
                n_init=n_init,
                random_state=random_state,
                **gmm_other_params,
            )
            gmm_model.fit(self.X_pca[i].values)
            gmm_models.append(gmm_model)

        self.gmm_models = gmm_models
        self.gmm_n_components_list = n_components_list

        return gmm_models

    def fit_predict_gmm(
        self,
        n_components_list,
        covariance_type="full",
        max_iter=2000,
        n_init=10,
        random_state=None,
        gmm_other_params={},
    ):
        """
        Fit GMM models with each day's data and predict labels for them.

        Parameters
        ----------
        n_components_list : list of int
            Each element corresponds to the number of components of the GMM model for each day.
            Passed to the 'n_components' parameter of the GaussianMixture class.
            
        covariance_type : {'full', 'tied', 'diag', 'spherical'}, optional
            String describing the type of covariances parameters to use, by default "full"
            Passed to the 'covariance_type' parameter of the GaussianMixture class.
            
        max_iter : int, optional
            The number of EM iterations to perform, by default 2000
            Passed to the 'max_iter' parameter of the GaussianMixture class.
            
        n_init : int, optional
            The number of initializations to perform, by default 10
            Passed to the 'n_init' parameter of the GaussianMixture class.
            
        random_state : int, RandomState instance or None, optional
            Controls the random seed given at each GMM model initialization, by default None
            Passed to the 'random_state' parameter of the GaussianMixture class.
            
        gmm_other_params : dict, optional
            Other parameters for GMM, by default {}

        Returns
        -------
        list of GaussianMixture instances
            The length of the list is the same as the number of days.
            Each element is a GMM instance fitted to the corresponding day's data.
            
        list of np.ndarray
            List of GMM labels.
            Each element is the predicted labels for the corresponding day's data.
        """

        if self.verbose:
            print(
                "Fitting GMM models with each day's data and predicting labels for them..."
            )
        gmm_models, gmm_labels = [], []
        for i in (
            tqdm(range(len(self.X_pca))) if self.verbose else range(len(self.X_pca))
        ):
            gmm_model = GaussianMixture(
                n_components_list[i],
                covariance_type=covariance_type,
                max_iter=max_iter,
                n_init=n_init,
                random_state=random_state,
                **gmm_other_params,
            )
            gmm_labels.append(gmm_model.fit_predict(self.X_pca[i].values))
            gmm_models.append(gmm_model)

        self.gmm_n_components_list = n_components_list
        self.gmm_models = gmm_models
        self.gmm_labels = gmm_labels
        self.gmm_labels_modified = gmm_labels

        return gmm_models, gmm_labels

    def predict_gmm_label(self, X_item, gmm_model):
        return gmm_model.predict(X_item.values)

    def predict_gmm_labels(self, X, gmm_models):
        gmm_labels = [
            self.predict_gmm_label(X[i], gmm_models[i]) for i in range(len(X))
        ]
        if self.gmm_labels is None:
            self.gmm_labels = gmm_labels
            self.gmm_labels_modified = gmm_labels
        return gmm_labels

    def _plot_gmm_predictions(
        self,
        X_item,
        x_range,
        y_range,
        figure_labels=None,
        gmm_label=None,
        gmm_n_components=None,
        cmap="plasma",
    ):
        if gmm_label is None:
            plt.scatter(X_item.values[:, 0], X_item.values[:, 1], s=1.0, alpha=0.8)
        else:
            plt.scatter(
                X_item.values[:, 0],
                X_item.values[:, 1],
                c=gmm_label,
                alpha=0.5,
                cmap=plt.get_cmap(cmap, gmm_n_components),
            )
            plt.colorbar(ticks=range(gmm_n_components), label="cluster")
            plt.clim(-0.5, gmm_n_components - 0.5)

        if figure_labels is not None:
            plt.xlabel(figure_labels[0])
            plt.ylabel(figure_labels[1])

        plt.xlim(x_range)
        plt.ylim(y_range)

    def plot_gmm_predictions(
        self,
        mode="pca",
        figure_labels=None,
        x_range=None,
        y_range=None,
        figure_titles_without_gmm=None,
        figure_titles_with_gmm=None,
        plot_gmm_means=False,
        cmap="plasma",
        save=False,
        save_paths=None,
    ):
        """
        Plot GMM predictions.
        Output images for the number of days.
        Each image contains two subplots: left one is in one color and right one is 
        colored by GMM labels.

        Parameters
        ----------
        mode : {'pca', 'umap'}, optional
            The space to plot the GMM predictions, by default "pca"
        
        figure_labels : list or tuple of str of shape (2,), optional
            X and Y axis labels, by default None
            If None, the first two columns of the input data will be used.

        x_range : list or tuple of float of shape (2,), optional
            Restrict the X axis range, by default None
            If None, the range will be automatically determined to include all data points.

        y_range : list or tuple of float of shape (2,), optional
            Restrict the Y axis range, by default None
            If None, the range will be automatically determined to include all data points.
            
        figure_titles_without_gmm : list or tuple of str of shape (n_days,), optional
            List of figure titles of left subplots, by default None
    
        figure_titles_with_gmm : list or tuple of str of shape (n_days,), optional
            List of figure titles of right subplots, by default None

        plot_gmm_means : bool, optional
            If True, plot GMM mean points on the right subplots, by default False

        cmap : str, optional
            String of matplolib colormap name, by default "plasma"

        save : bool, optional
            If True, save the output images, by default False

        save_paths : list or tuple of str of shape (n_days), optional
            List of paths to save the output images, by default None
            If None, the images will be saved as './GMM_preds_{i + 1}.png'.

        Raises
        ------
        ValueError
            When 'mode' is not 'pca' or 'umap'.
        """

        if mode not in ["pca", "umap"]:
            raise ValueError("The parameter 'mode' should be 'pca' or 'umap'.")

        X = self.X_pca if mode == "pca" else self.X_umap

        if x_range is None:
            x_min = min([np.min(df.iloc[:, 0].values) for df in X])
            x_max = max([np.max(df.iloc[:, 0].values) for df in X])
            x_range = (x_min - (x_max - x_min) / 10, x_max + (x_max - x_min) / 10)

        if y_range is None:
            y_min = min([np.min(df.iloc[:, 1].values) for df in X])
            y_max = max([np.max(df.iloc[:, 1].values) for df in X])
            y_range = (y_min - (y_max - y_min) / 10, y_max + (y_max - y_min) / 10)

        if figure_titles_without_gmm is None:
            figure_titles_without_gmm = [
                f"Figure {i + 1} - wighout GMM" for i in range(len(X))
            ]

        if figure_titles_with_gmm is None:
            figure_titles_with_gmm = [
                f"Figure {i + 1} - with GMM" for i in range(len(X))
            ]

        if figure_labels is None:
            figure_labels = X[0].columns[:2].values

        if save and save_paths is None:
            save_paths = [f"./GMM_preds_{i + 1}.png" for i in range(len(X))]

        for i, X_item in enumerate(X):
            plt.figure(figsize=(20, 8))

            plt.subplot(1, 2, 1)
            self._plot_gmm_predictions(X_item, x_range, y_range, figure_labels)
            plt.title(figure_titles_without_gmm[i], fontsize=20)

            plt.subplot(1, 2, 2)
            gmm_model = self.gmm_models[i]

            if plot_gmm_means:
                if mode == "pca":
                    means = gmm_model.means_
                else:
                    means = self.umap_model.transform(gmm_model.means_)
                for k in range(self.gmm_n_components_list[i]):
                    plt.plot(
                        means[k][0],
                        means[k][1],
                        "ro",
                        markersize=12,
                        markeredgewidth=3,
                        markeredgecolor="white",
                    )

            self._plot_gmm_predictions(
                X_item,
                x_range,
                y_range,
                figure_labels,
                self.gmm_labels_modified[i],
                self.gmm_n_components_list[i],
                cmap,
            )
            plt.title(figure_titles_with_gmm[i], fontsize=20)

            if save:
                plt.savefig(save_paths[i], dpi=600)
            plt.show()

    def _interpolation_contour(
        self, gmm_source, gmm_target, t, x_range, y_range, cmap="rainbow"
    ):
        K_0, K_1 = gmm_source.means_.shape[0], gmm_target.means_.shape[0]

        solution = self.calculate_solution(gmm_source, gmm_target)
        pit = solution.reshape(K_0 * K_1, 1).T

        mut, St = self.calculate_mut_st(gmm_source, gmm_target, t)
        x, y = np.meshgrid(
            np.linspace(x_range[0], x_range[1], 1000),
            np.linspace(y_range[0], y_range[1], 1000),
        )
        flattened = np.array([x.ravel(), y.ravel()]).T
        z = self.gaussian_mixture_density(mut[:, 0:2], St[:, 0:2, 0:2], pit, flattened)
        z = z.reshape(x.shape)
        max_z = np.max(z)
        min_z = np.min(z)
        img = plt.contour(x, y, z, np.linspace(min_z - 1e-9, max_z, 20), cmap=cmap)

        collections = []

        img.set_visible(False)

        fcs = img.get_facecolor()
        ecs = img.get_edgecolor()
        lws = img.get_linewidth()
        lss = img.get_linestyle()

        for i, path in enumerate(img.get_paths()):
            pc = collect.PathCollection(
                    [path] if len(path.vertices) else [],
                    alpha=img.get_alpha(),
                    antialiaseds=img.get_antialiased(),
                    transform=img.get_transform(),
                    zorder=img.get_zorder(),
                    label="_nolegend_",
                    facecolor=fcs[i] if len(fcs) else "none",
                    edgecolor=ecs[i] if len(ecs) else "none",
                    linewidths=[lws[i]],
                    linestyles=[lss[i]],
                )
            collections.append(pc)

        for collection in collections:
            img.axes.add_collection(collection)

        return(collections)

    def animatie_interpolated_distribution(
        self,
        x_range=None,
        y_range=None,
        interpolate_interval=11,
        cmap="gnuplot2",
        save=False,
        save_path=None,
    ):
        """
        Export an animation of the interpolated distribution between GMM models.

        Parameters
        ----------
        x_range : list or tuple of float of shape (2,), optional
            Restrict the X axis range, by default None

        y_range : list or tuple of float of shape (2,), optional
            Restrict the Y axis range, by default None

        interpolate_interval : int, optional
            The number of frames to interpolate between two timepoints, by default 11
            This is the total number of frames at both timepoints and the number of frames
            between these.
            Note that both ends are included.

        cmap : str, optional
            String of matplolib colormap name, by default "gnuplot2"

        save : bool, optional
            If True, save the output animation, by default False

        save_path : str, optional
            Path to save the output animation, by default None
            If None, the animation will be saved as './cell_state_video.gif'
        """

        if save and save_path is None:
            save_path = "./cell_state_video.gif"

        if x_range is None:
            x_min = min([np.min(df.iloc[:, 0].values) for df in self.X_pca])
            x_max = max([np.max(df.iloc[:, 0].values) for df in self.X_pca])
            x_range = (x_min - (x_max - x_min) / 10, x_max + (x_max - x_min) / 10)

        if y_range is None:
            y_min = min([np.min(df.iloc[:, 1].values) for df in self.X_pca])
            y_max = max([np.max(df.iloc[:, 1].values) for df in self.X_pca])
            y_range = (y_min - (y_max - y_min) / 10, y_max + (y_max - y_min) / 10)

        ims = []

        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(len(self.day_names) - 1):
            t = np.linspace(0, 1, interpolate_interval)
            print(
                f"Interpolating between {self.day_names[i]} and {self.day_names[i + 1]}..."
            )
            frames = range(int(i > 0), interpolate_interval)
            for j in tqdm(frames) if self.verbose else frames:
                if i != 0 and j == 0:
                    continue
                im = self._interpolation_contour(
                    self.gmm_models[i],
                    self.gmm_models[i + 1],
                    t[j],
                    x_range,
                    y_range,
                    cmap,
                )
                title = ax.text(
                    0.5,
                    1.0,
                    f"{self.day_names[i]}, t = {t[j]:.1f}",
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=20,
                )
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                ims.append(im + [title])

        if self.verbose:
            print("Creating animation...")
        anim = animation.ArtistAnimation(fig, ims, interval=100, repeat=False)
        if is_notebook():
            display(HTML(anim.to_jshtml()))
        else:
            plt.show()

        if save:
            anim.save(save_path, writer="pillow")

        plt.close()
    
    def _get_cell_state_edge_list_old(self, cluster_names, thresh):
        node_source_target_combinations, edge_colors_based_on_source = [], []
        for i in range(len(self.gmm_n_components_list) - 1):
            current_combinations = [
                x for x in itertools.product(cluster_names[i], cluster_names[i + 1])
            ]
            node_source_target_combinations += current_combinations
            edge_colors_based_on_source += [i for _ in range(len(current_combinations))]
        cell_state_edge_list = pd.DataFrame(
            node_source_target_combinations, columns=["source", "target"]
        )
        cell_state_edge_list["edge_colors"] = edge_colors_based_on_source
        cell_state_edge_list["edge_weights"] = list(
            itertools.chain.from_iterable(
                list(
                    itertools.chain.from_iterable(
                        self.calculate_normalized_solutions(self.gmm_models)
                    )
                )
            )
        )
        cell_state_edge_list = cell_state_edge_list[
            cell_state_edge_list["edge_weights"] > thresh
        ]

        return cell_state_edge_list

    def _get_gmm_node_weights_flattened(self):
        node_weights = [
            self.gmm_models[i].weights_ for i in range(len(self.gmm_models))
        ]
        node_weights = list(itertools.chain.from_iterable(node_weights))
        return node_weights

    def _get_day_order_of_each_node(self):
        day_names_of_each_node = []
        for i, gmm_n_components in enumerate(self.gmm_n_components_list):
            day_names_of_each_node += [i] * gmm_n_components
        return day_names_of_each_node

    def _get_nlargest_gene_indices(self, row, num=10):
        nlargest = row.nlargest(num)
        return nlargest.index

    def _get_nsmallest_gene_indices(self, row, num=10):
        nsmallest = row.nsmallest(num)
        return nsmallest.index

    def _get_fold_change(self, gene_values, source, target):
        fold_change = pd.Series(
            gene_values.T[target] - gene_values.T[source], index=gene_values.T.index
        )
        fold_change = fold_change.sort_values(ascending=False)
        return fold_change
    
    def _calculate_source_merged_edge_weights(self, df):
        node_weights = self._get_gmm_node_weights_flattened()
        source_node_weights = df.apply(lambda row: node_weights[int(row["source_cluster"])], axis=1)
        edge_weights_weighted_by_source = source_node_weights * df["edge_weights"]
        return pd.Series(
            {
                "edge_colors": df["edge_colors"].min(),
                "edge_weights": sum(edge_weights_weighted_by_source) / sum(source_node_weights)
            }
        )

    def _get_cell_state_edge_list(self, node_ids, merge_clusters_by_name, thresh, require_parent):
        node_source_target_combinations = []
        day_source_target_combinations = []
        edge_colors_based_on_source = []

        cluster_nums = self.gmm_n_components_list
        cluster_start_indexes = [0] + [sum(cluster_nums[:i+1]) for i in range(len(cluster_nums))] + [sum(cluster_nums)]
        
        for i in range(len(cluster_nums) - 1):
            current_combinations = list(itertools.product(
                list(range(cluster_start_indexes[i], cluster_start_indexes[i+1])),
                list(range(cluster_start_indexes[i+1], cluster_start_indexes[i+2]))
            ))            
            node_source_target_combinations += current_combinations
            day_source_target_combinations += [(i, i+1)] * len(current_combinations)
            edge_colors_based_on_source += [i for _ in range(len(current_combinations))]

        cell_state_edges = pd.DataFrame(
            node_source_target_combinations, 
            columns=["source_cluster", "target_cluster"]
        ).join(pd.DataFrame(
            day_source_target_combinations,
            columns=["source_day", "target_day"]
        ))
        cell_state_edges["edge_colors"] = edge_colors_based_on_source
        cell_state_edges["edge_weights"] = list(
            itertools.chain.from_iterable(
                itertools.chain.from_iterable(
                    self.calculate_normalized_solutions(self.gmm_models)
                )
            )
        )
        if merge_clusters_by_name:
            cell_state_edges["target"] = [node_ids[i] for i in cell_state_edges["target_cluster"]]
            cell_state_edges = cell_state_edges.groupby(
                ["source_cluster", "source_day", "target", "target_day"],as_index=False
            ).agg({"edge_colors": "min", "edge_weights": "sum"})

            cell_state_edges["source"] = [node_ids[i] for i in cell_state_edges["source_cluster"]]
            cell_state_edges = cell_state_edges.groupby(
                ["source", "source_day", "target", "target_day"], as_index=False
            ).apply(self._calculate_source_merged_edge_weights)
            cell_state_edges["edge_colors"] = cell_state_edges["edge_colors"].astype(int)

        else:
            cell_state_edges["source"] = [node_ids[i] for i in cell_state_edges["source_cluster"]]
            cell_state_edges["target"] = [node_ids[i] for i in cell_state_edges["target_cluster"]]

        filtered_cell_state_edges = cell_state_edges[cell_state_edges["edge_weights"] >= thresh]

        if require_parent:
            unique_target_node_ids_flattened = node_ids[1:]
            for id in unique_target_node_ids_flattened:
                if not id in filtered_cell_state_edges["target"].values:
                    current_df = cell_state_edges[cell_state_edges["target"] == id]
                    max_weight = current_df["edge_weights"].max()
                    additional_rows = current_df[current_df["edge_weights"] == max_weight]
                    filtered_cell_state_edges = pd.concat([filtered_cell_state_edges, additional_rows])
                    
        return filtered_cell_state_edges
    
    def _generate_merged_node_ids(self, cluster_names):
        cluster_days = self._get_day_order_of_each_node()
        cluster_names_flattened = list(itertools.chain.from_iterable(cluster_names))
        node_ids = []
        cluster_id_dict_list = []
        accum_n_node = 0
        for day_cluster_names in cluster_names:
            day_cluster_set = sorted(list(set(day_cluster_names)), key=day_cluster_names.index)
            n_day_node = len(day_cluster_set)
            cluster_id_dict = dict(zip(day_cluster_set, range(accum_n_node, accum_n_node + n_day_node)))
            cluster_id_dict_list.append(cluster_id_dict)
            accum_n_node += n_day_node
        node_ids = [cluster_id_dict_list[day][name] for day, name in zip(cluster_days, cluster_names_flattened)]
        return node_ids

    def merge_cluster_names_by_pathway(
        self,
        last_day_cluster_names,
        n_merge_iter=None,
        merge_method="pattern",
        threshold=0.05,
        n_clusters_list=None,
        **kmeans_kwargs
    ):
        """
        Merge cluster names based on cell state graph pathways.

        Parameters
        ----------
        last_day_cluster_names : list of str
            Cluster names for the last day.
            Clusters with the same name will be merged.

            The length of the list should be equal to the number of clusters in the last day.
        
        n_merge_iter : int, optional
            Number of preceding days to trace back and merge cluster names, starting from
            the last day, by default (the number of days - 1).
            
            Must be an integer in the range from 1 to (the number of days - 1).

        merge_method : {'pattern', 'kmeans'}, optional
            Method to merge nodes, by default 'pattern'.

            * 'pattern': Merges nodes that share the same connection pattern to the next day's nodes.
            * 'kmeans': Merges nodes based on the edge weights to the next day's nodes using K-Means.
        
        threshold : float, optional
            Threshold to filter edges, by default 0.05.
            Edges with weights below this value are ignored.

            This parameter is used only when `merge_method` is 'pattern'.

        n_clusters_list : list of int, optional
            List specifying the number of merged clusters for each day.

            The length of the list must equal to the number of days or (the number of days - 1).
            If None, defaults to the minimum of (original cluster count, 4) for each day.

            This parameter is used only when `merge_method` is 'kmeans'.

        \*\*kmeans_kwargs : dict
            Arbitrary keyword arguments passed to sklearn.cluster.KMeans.

            This parameter is used only when `merge_method` is 'kmeans'.

        Returns
        -------
        list of list of str
            Merged cluster names for each day.

        Raises
        ------
        ValueError
            This error is raised in the following cases:

            * When 'n_merge_iter' is not an integer within the valid range (1 to number of days - 1).
            * When 'merge_method' is not one of 'pattern' or 'kmeans'.
        """

        if (n_merge_iter is not None) and (not n_merge_iter in list(range(1, len(self.day_names)))):
            raise ValueError(f"The parameter 'n_merge_iter' should be an integer from 1 to {len(self.day_names) - 1}.")

        if not merge_method in ["pattern", "kmeans"]:
            raise ValueError("The parameter 'merge_method' should be 'pattern' or 'kmeans'")
        
        if n_merge_iter is None:
            n_merge_iter = len(self.day_names) - 1

        if merge_method == "kmeans" and n_clusters_list == None:
            n_clusters_list = [min(gmm_n_components, 4) for gmm_n_components in self.gmm_n_components_list]

        n_days = len(self.day_names)
        cluster_names = self.generate_cluster_names_with_day()
        cluster_names[-1] = last_day_cluster_names

        node_ids = self._generate_merged_node_ids(cluster_names)
        cluster_days = self._get_day_order_of_each_node()
        node_ids_by_day = [[] for _ in range(len(self.day_names))]
        for day, node_id in zip(cluster_days, node_ids):
            node_ids_by_day[day].append(node_id)

        for i in range(n_merge_iter):    
            source_node_ids = node_ids_by_day[-(i+2)]
            unique_target_node_ids = list(set(node_ids_by_day[-(i+1)]))
            node_ids = list(itertools.chain.from_iterable(node_ids_by_day))

            if merge_method == "pattern":
                cell_state_edges = self._get_cell_state_edge_list(node_ids, True, threshold, False)
                current_day_df = cell_state_edges[cell_state_edges["source"].isin(source_node_ids)]

                target_combination_df = current_day_df.groupby("source", as_index=False).apply(
                    lambda df: pd.Series(
                        [(target in df["target"].tolist()) for target in unique_target_node_ids],
                        index=unique_target_node_ids
                    )
                )

                clusters_groupby_target = []
                target_combination_df.groupby(unique_target_node_ids, as_index=False).apply(
                    lambda df: clusters_groupby_target.append(df["source"].tolist())
                )

                new_source_cluster_ids = [None] * len(source_node_ids)
                node_id_base = min(source_node_ids)
                for group_num, group in enumerate(clusters_groupby_target):
                    for cluster in group:
                        for index, id in enumerate(source_node_ids):
                            if cluster == id:
                                new_source_cluster_ids[index] = node_id_base + group_num

            if merge_method == "kmeans":
                cell_state_edges = self._get_cell_state_edge_list(node_ids, True, 0, False)
                current_day_df = cell_state_edges[cell_state_edges["source"].isin(source_node_ids)]

                source_target_df = pd.DataFrame(
                    [
                        [current_day_df[(current_day_df["source"] == source) & (current_day_df["target"] == target)]["edge_weights"].values[0]
                        for target in unique_target_node_ids]
                        for source in source_node_ids
                    ],
                    columns=unique_target_node_ids,
                    index=source_node_ids
                )
                
                kmeans_model = KMeans(n_clusters=n_clusters_list[n_days - (i+2)], **kmeans_kwargs).fit(source_target_df)
                node_id_base = min(source_node_ids)
                new_source_cluster_ids = []
                for group_num in kmeans_model.labels_:
                    new_source_cluster_ids.append(node_id_base + group_num)

            node_ids_by_day[-(i+2)] = new_source_cluster_ids

        day_names = self.day_names
        new_cluster_names = []
        for day_name, day_node_ids in zip(day_names[:-1], node_ids_by_day[:-1]):
            node_id_base = min(day_node_ids)
            new_day_cluster_names = []
            for node_id in day_node_ids:
                new_day_cluster_names.append(f"{day_name}-{node_id - node_id_base}")
            new_cluster_names.append(new_day_cluster_names)
        new_cluster_names.append(last_day_cluster_names)

        return new_cluster_names

    def _merge_nodes(self, node_info_df):
        node_info_df = node_info_df.set_index("id")

        merged_df = node_info_df.groupby(level=0).agg({
            "day": "min",
            "weight": "sum",
            "cluster_gmm_list": lambda x: sum(x.values.tolist(), start=[])
        })

        xpos = node_info_df["xpos"]
        ypos = node_info_df["ypos"]
        weights = node_info_df["weight"]

        merged_df["xpos"] = (xpos * weights).groupby(level=0).sum() / weights.groupby(level=0).sum()
        merged_df["ypos"] = (ypos * weights).groupby(level=0).sum() / weights.groupby(level=0).sum()

        return merged_df

    def _get_up_regulated_genes(self, gene_values, G, num=10):
        df_upgenes = pd.DataFrame([])
        for edge in G.edges():
            fold_change = self._get_fold_change(
                gene_values,
                edge[0],
                edge[1],
            )
            upgenes = pd.DataFrame(
                self._get_nlargest_gene_indices(fold_change, num=num).values,
                columns=[f"{edge[0]}{edge[1]}"],
            ).T
            df_upgenes = pd.concat([df_upgenes, upgenes])
        return df_upgenes

    def _get_down_regulated_genes(self, gene_values, G, num=10):
        df_downgenes = pd.DataFrame([])
        for edge in G.edges():
            fold_change = self._get_fold_change(
                gene_values,
                edge[0],
                edge[1],
            )
            downgenes = pd.DataFrame(
                self._get_nsmallest_gene_indices(fold_change, num=num).values,
                columns=[f"{edge[0]}{edge[1]}"],
            ).T
            df_downgenes = pd.concat([df_downgenes, downgenes])
        return df_downgenes
    
    def make_cell_state_graph(
        self,
        cluster_names,
        mode="pca",
        threshold=0.05,
    ):
        """
        .. warning::
            ``make_cell_state_graph()`` was deprecated in version 0.3.0 and will be removed in future versions.
            Use ``make_cell_state_graph_object()`` instead.
        
        Compute cell state graph and build a networkx graph object.

        Parameters
        ----------
        cluster_names : 2D list of str
            1st dimension is the number of days, 2nd dimension is the number of gmm components
            in each day.
            Can be generaged by 'generate_cluster_names' method.

        mode : {'pca', 'umap'}, optional
            The space to build the cell state graph, by default "pca"
            
        threshold : float, optional
            Threshold to filter edges, by default 0.05
            Only edges with edge_weights greater than this threshold will be included.

        Returns
        -------
        nx.classes.digraph.DiGraph
            Networkx graph object of the cell state graph

        Raises
        ------
        ValueError
            When 'mode' is not 'pca' or 'umap'.
        """

        warnings.warn(
            "'make_cell_state_graph()' was deprecated and will be removed in future versions.\n"
            "Use 'make_cell_state_graph_object()' instead.",
            FutureWarning,
        )

        if mode not in ["pca", "umap"]:
            raise ValueError("The parameter 'mode' should be 'pca' or 'umap'.")

        gmm_means_flattened = np.array(
            list(itertools.chain.from_iterable(self.get_gmm_means()))
        )
        if mode == "umap":
            gmm_means_flattened = self.umap_model.transform(gmm_means_flattened)

        cell_state_edge_list = self._get_cell_state_edge_list_old(cluster_names, threshold)
        G = nx.from_pandas_edgelist(
            cell_state_edge_list,
            source="source",
            target="target",
            edge_attr=["edge_weights", "edge_colors"],
            create_using=nx.DiGraph,
        )
        node_info = pd.DataFrame(
            self._get_gmm_node_weights_flattened(),
            index=list(itertools.chain.from_iterable(cluster_names)),
            columns=["node_weights"],
        )
        node_info["xpos"] = gmm_means_flattened.T[0]
        node_info["ypos"] = gmm_means_flattened.T[1]
        node_info["node_days"] = self._get_day_order_of_each_node()
        if self.gmm_label_converter is None:
            node_info["cluster_gmm"] = list(
                itertools.chain.from_iterable(
                    [
                        list(range(n_components))
                        for n_components in self.gmm_n_components_list
                    ]
                )
            )
        else:
            node_info["cluster_gmm"] = list(
                itertools.chain.from_iterable(self.gmm_label_converter)
            )

        node_sortby_weight = (
            node_info.reset_index()
            .groupby("node_days")
            .apply(
                lambda x: x.sort_values("node_weights", ascending=False),
                include_groups=False,
            )
        )
        node_sortby_weight = node_sortby_weight.reset_index()
        node_info = pd.DataFrame(
            node_sortby_weight.values, columns=node_sortby_weight.columns
        )
        node_info["cluster_weight"] = list(
            itertools.chain.from_iterable(
                [
                    list(range(n_components))
                    for n_components in self.gmm_n_components_list
                ]
            )
        )
        node_info.set_index("index", inplace=True)

        for row in node_info.itertuples():
            G.add_node(
                row.Index,
                weight=row.node_weights,
                day=row.node_days,
                pos=(row.xpos, row.ypos),
                cluster_gmm=row.cluster_gmm,
                cluster_weight=row.cluster_weight,
            )

        return G

    def make_cell_state_graph_object(
        self,
        cluster_names=None,
        mode="pca",
        threshold=0.05,
        merge_clusters_by_name=False,
        x_reverse=False,
        y_reverse=False,
        require_parent=False,
    ):
        """
        Compute cell state graph and build a ``CellStateGraph`` object.

        Parameters
        ----------
        cluster_names : 2D list of str, optional
            Cluster names for each GMM cluster in each day.
            1st dimension is the number of days, 2nd dimension is the number of gmm components
            in each day.

            If merge_clusters_by_name is True, clusters with the same name will be merged.

            If None, generated by ``generate_cluster_names_with_day()`` method.
            
        mode : {'pca', 'umap'}, optional
            The space to build the cell state graph, by default 'pca'

        threshold : float, optional
            Threshold to filter edges, by default 0.05
            Only edges with edge_weights greater than this threshold will be included.

        merge_clusters_by_name : bool, optional
            If True, clusters with the same name will be merged, by default False
        
        x_reverse : bool, optional
            If True, reverse the X axis direction, by default False

        y_reverse : bool, optional
            If True, reverse the Y axis direction, by default False
        
        require_parent : bool, optional
            If True, ensure that each cluster in the target day has at least one incoming
            edge from the source day, by default False
        
        Returns
        -------
        scegot.CellStateGraph
            ``scegot.CellStateGraph`` object of the cell state graph

        Raises
        ------
        ValueError
            This error is raised in the following cases:

            * When 'mode' is not 'pca' or 'umap'.
            * When the length of 'cluster_names' is not the same as the number of days.
            * When the length of the second dimension of 'cluster_names' is not the same
              as the number of GMM components in each day.
        """

        if mode not in ["pca", "umap"]:
            raise ValueError("The parameter 'mode' should be 'pca' or 'umap'.")

        if cluster_names:
            if (len(cluster_names) != len(self.day_names)):
                raise ValueError(
                    "Size of the first dimension of 'cluster_names' should be the same "
                    "as the number of days."
                )
            if not [len(day_cluster_names) for day_cluster_names in cluster_names] == self.gmm_n_components_list:
                raise ValueError(
                    "Size of the second dimension of 'cluster_names' should be the same "
                    "as the number of GMM components in each day."
                )
        
        if cluster_names is None:
            cluster_names = self.generate_cluster_names_with_day()

        if merge_clusters_by_name:
            node_ids = self._generate_merged_node_ids(cluster_names)
        else:
            node_ids = list(range(sum(self.gmm_n_components_list)))

        gmm_means_flattened = np.array(
            list(itertools.chain.from_iterable(self.get_gmm_means()))
        )
        if mode == "umap":
            gmm_means_flattened = self.umap_model.transform(gmm_means_flattened)

        cell_state_edge_list = self._get_cell_state_edge_list(node_ids, merge_clusters_by_name, threshold, require_parent)
        cell_state_edge_list.rename(columns={"edge_weights": "weight", "edge_colors": "color"}, inplace=True)
        G = nx.from_pandas_edgelist(
            cell_state_edge_list,
            source="source",
            target="target",
            edge_attr=["weight", "color"],
            create_using=nx.DiGraph,
        )

        node_info = pd.DataFrame()

        cluster_days = self._get_day_order_of_each_node()

        node_info["id"] = node_ids
        node_info["day"] = cluster_days
        node_info["weight"] = self._get_gmm_node_weights_flattened()

        if x_reverse:
            node_info["xpos"] = gmm_means_flattened.T[0] * (-1)
        else:
            node_info["xpos"] = gmm_means_flattened.T[0]
        if y_reverse:
            node_info["ypos"] = gmm_means_flattened.T[1] * (-1)
        else: 
            node_info["ypos"] = gmm_means_flattened.T[1]

        if self.gmm_label_converter is None:
            cluster_gmms = list(
                itertools.chain.from_iterable(
                    [list(range(n_components)) for n_components in self.gmm_n_components_list]
                )
            )
        else:
            cluster_gmms = list(
                itertools.chain.from_iterable(self.gmm_label_converter)
            )
        node_info["cluster_gmm_list"] = [[x] for x in cluster_gmms]

        if merge_clusters_by_name:
            merged_node_info = self._merge_nodes(node_info)
        else:
            merged_node_info = node_info

        cluster_weights = merged_node_info.groupby("day")["weight"].rank(ascending=False).astype(int) - 1
        merged_node_info["cluster_weight"] = cluster_weights

        for row in merged_node_info.itertuples():
            G.add_node(
                row.Index,
                weight=row.weight,
                day=row.day,
                pos=(row.xpos, row.ypos),
                cluster_gmm_list=row.cluster_gmm_list,
                cluster_weight=row.cluster_weight,
            )

        graph = CellStateGraph(
            G,
            scegot=self,
            threshold=threshold,
            mode=mode,
            cluster_names=copy.deepcopy(cluster_names),
            node_ids=node_ids,
            merge_clusters_by_name=merge_clusters_by_name,
            x_reverse=x_reverse,
            y_reverse=y_reverse,
            require_parent=require_parent,
        )

        return graph

    def _plot_cell_state_graph(
        self,
        G,
        nodes_up_gene,
        nodes_down_gene,
        edges_up_gene,
        edges_down_gene,
        save,
        save_path,
    ):
        tail_list = []
        head_list = []
        color_list = []
        trace_recode = []

        day_num = len(self.day_names)

        colors = plt.cm.inferno(np.linspace(0, 1, day_num + 2))
        for edge in G.edges():
            x_0, y_0 = G.nodes[edge[0]]["pos"]
            x_1, y_1 = G.nodes[edge[1]]["pos"]
            tail_list.append((x_0, y_0))
            head_list.append((x_1, y_1))
            weight = G.edges[edge]["edge_weights"] * 25
            color = colors[G.edges[edge]["edge_colors"] + 1]

            color_list.append(f"rgb({color[0]},{color[1]},{color[2]})")

            edge_trace = go.Scatter(
                x=tuple([x_0, x_1, None]),
                y=tuple([y_0, y_1, None]),
                mode="lines",
                line={"width": weight},
                line_color=f"rgb({color[0]},{color[1]},{color[2]})",
                line_shape="spline",
                opacity=0.4,
            )

            trace_recode.append(edge_trace)

        middle_hover_trace = go.Scatter(
            x=[],
            y=[],
            hovertext=[],
            mode="markers",
            textposition="top center",
            hoverinfo="text",
            marker={
                "size": 20,
                "color": [edge["edge_colors"] + 1 for edge in G.edges.values()],
            },
            opacity=0,
        )

        for edge in G.edges():
            x_0, y_0 = G.nodes[edge[0]]["pos"]
            x_1, y_1 = G.nodes[edge[1]]["pos"]
            from_to = str(edge[0]) + str(edge[1])
            hovertext = (
                f"up_genes: {', '.join(edges_up_gene.T[from_to].values)}<br>"
                f"down_genes: {', '.join(edges_down_gene.T[from_to].values)}"
            )
            middle_hover_trace["x"] += tuple([(x_0 + x_1) / 2])
            middle_hover_trace["y"] += tuple([(y_0 + y_1) / 2])
            middle_hover_trace["hovertext"] += tuple([hovertext])
            trace_recode.append(middle_hover_trace)

        arrows = [
            go.layout.Annotation(
                dict(
                    x=head[0],
                    y=head[1],
                    showarrow=True,
                    xref="x",
                    yref="y",
                    arrowcolor=color,
                    arrowsize=2,
                    arrowwidth=2,
                    ax=tail[0],
                    ay=tail[1],
                    axref="x",
                    ayref="y",
                    arrowhead=1,
                )
            )
            for head, tail, color in zip(head_list, tail_list, color_list)
        ]

        node_x = []
        node_y = []
        node_hover_trace = go.Scatter(
            x=[],
            y=[],
            hovertext=[],
            mode="markers",
            textposition="top center",
            hoverinfo="text",
            marker={
                "size": 20,
                "color": [node["day"] + 1 for node in G.nodes.values()],
            },
            opacity=0,
        )

        for node in G.nodes():
            x, y = G.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)
            hovertext = (
                f"largest_genes: {', '.join(nodes_up_gene.T[node].values)}<br>"
                f"smallest_genes: {', '.join(nodes_down_gene.T[node].values)}"
            )
            node_hover_trace["x"] += tuple([x])
            node_hover_trace["y"] += tuple([y])
            node_hover_trace["hovertext"] += tuple([hovertext])
            trace_recode.append(node_hover_trace)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=list(G.nodes()),
            textposition="top center",
            mode="markers+text",
            hoverinfo="text",
            marker=dict(line_width=2),
        )

        node_trace.marker.color = [node["day"] for node in G.nodes.values()]
        node_trace.marker.size = [node["weight"] * 140 for node in G.nodes.values()]
        trace_recode.append(node_trace)

        fig = go.Figure(
            data=trace_recode, layout=go.Layout(showlegend=False, hovermode="closest")
        )

        fig.update_layout(annotations=arrows)

        fig.update_layout(width=1000, height=800, title="Cell state graph")
        fig.show()

        if save:
            fig.write_image(save_path)

    def plot_cell_state_graph(
        self,
        G,
        cluster_names,
        tf_gene_names=None,
        tf_gene_pick_num=5,
        save=False,
        save_path=None,
    ):
        """
        .. warning::
            ``scEGOT.plot_cell_state_graph()`` was deprecated in version 0.3.0 and will be removed in future versions.
            Use ``CellStateGraph.plot_cell_state_graph()`` instead.
        
        Plot the cell state graph with the given graph object.

        Parameters
        ----------
        G : nx.classes.digraph.DiGraph
            Networkx graph object of the cell state graph.

        cluster_names : list of list of str
            1st dimension is the number of days, 2nd dimension is the number of gmm components
            of each day.
            Can be generaged by 'generate_cluster_names' method.

        tf_gene_names : list of str, optional
            List of transcription factor gene names to use, by default None
            If None, all gene names (self.gene_names) will be used.
            You can pass on any list of gene names you want to use, not limited to TF genes.

        tf_gene_pick_num : int, optional
            The number of genes to show in each node and edge, by default 5

        save : bool, optional
            If True, save the output image, by default False

        save_path : _type_, optional
            Path to save the output image, by default None
            If None, the image will be saved as './cell_state_graph.png'
        """
        
        warnings.warn(
            "'scEGOT.plot_cell_state_graph()' was deprecated and will be removed in future versions.\n"
            "Use 'CellStateGraph.plot_cell_state_graph()' instead.",
            FutureWarning,
        )

        if save and save_path is None:
            save_path = "./cell_state_graph.png"

        if tf_gene_names is None:
            gene_names_to_use = self.gene_names
        else:
            gene_names_to_use = tf_gene_names

        mean_gene_values_per_cluster = (
            self.get_positive_gmm_mean_gene_values_per_cluster(
                self.get_gmm_means(),
                list(itertools.chain.from_iterable(cluster_names)),
            )
        )
        mean_tf_gene_values_per_cluster = mean_gene_values_per_cluster.loc[
            :, mean_gene_values_per_cluster.columns.isin(gene_names_to_use)
        ]

        tf_nlargest = mean_tf_gene_values_per_cluster.T.apply(
            self._get_nlargest_gene_indices, num=tf_gene_pick_num
        ).T
        tf_nsmallest = mean_tf_gene_values_per_cluster.T.apply(
            self._get_nsmallest_gene_indices, num=tf_gene_pick_num
        ).T
        tf_nlargest.columns += 1
        tf_nsmallest.columns += 1

        tf_up_genes = self._get_up_regulated_genes(
            mean_tf_gene_values_per_cluster, G, num=tf_gene_pick_num
        )
        tf_down_genes = self._get_down_regulated_genes(
            mean_tf_gene_values_per_cluster, G, num=tf_gene_pick_num
        )
        tf_up_genes.columns += 1
        tf_down_genes.columns += 1

        self._plot_cell_state_graph(
            G,
            nodes_up_gene=tf_nlargest,
            nodes_down_gene=tf_nsmallest,
            edges_up_gene=tf_up_genes,
            edges_down_gene=tf_down_genes,
            save=save,
            save_path=save_path,
        )
    
    def plot_simple_cell_state_graph(
        self, G, layout="normal", order=None, save=False, save_path=None
    ):
        """
        .. warning::
            ``scEGOT.plot_simple_cell_state_graph()`` was deprecated in version 0.3.0 and will be removed in future versions.
            Use ``CellStateGraph.plot_simple_cell_state_graph()`` instead.
        
        Plot the cell state graph with the given graph object in a simple way.

        Parameters
        ----------
        G : nx.classes.digraph.DiGraph
            Networkx graph object of the cell state graph.

        layout : {'normal', 'hierarchy'}, optional
            The layout of the graph, by default "normal"
            When 'normal', the graph is plotted the same layout as the self.plot_cell_state_graph method.
            When 'hierarchy', the graph is plotted with the day on the x-axis and the cluster on the y-axis.

        order : {'weight', None}, optional
            Order of nodes along the y-axis, by default None
            This parameter is only used when 'layout' is 'hierarchy'.
            When 'weight', the nodes are ordered by the size of the nodes.
            When None, the nodes are ordered by the cluster number.

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './simple_cell_state_graph.png'

        Raises
        ------
        ValueError
            When 'layout' is not 'normal' or 'hierarchy', or 'order' is not None or 'weight'.
        """

        warnings.warn(
            "'scEGOT.plot_simple_cell_state_graph()' was deprecated and will be removed in future versions.\n"
            "Use 'CellStateGraph.plot_simple_cell_state_graph()' instead.",
            FutureWarning,
        )
        
        if layout not in ["normal", "hierarchy"]:
            raise ValueError("The parameter 'layout' should be 'normal or 'hierarchy'.")
        if order is not None and order != "weight":
            raise ValueError("The parameter 'order' should be None or 'weight'.")

        if save and save_path is None:
            save_path = "./simple_cell_state_graph.png"

        node_color = [node["day"] for node in G.nodes.values()]

        color_data = np.array([G.edges[edge]["edge_weights"] for edge in G.edges()])

        if layout == "normal":
            pos = {node: G.nodes[node]["pos"] for node in G.nodes()}
        else:
            pos = {}
            for node in G.nodes():
                if order is None:
                    pos[node] = (G.nodes[node]["day"], -G.nodes[node]["cluster_gmm"])
                else:
                    pos[node] = (G.nodes[node]["day"], -G.nodes[node]["cluster_weight"])
        fig, ax = plt.subplots(figsize=(12, 10))

        nx.draw(
            G,
            pos,
            node_size=[node["weight"] * 4500 for node in G.nodes.values()],
            node_color="white",
            edge_color="black",
            arrows=True,
            arrowsize=30,
            linewidths=2,
            ax=ax,
            width=6.0,
        )
        nx.draw(
            G,
            pos,
            node_size=[node["weight"] * 5000 for node in G.nodes.values()],
            node_color="white",
            edge_color="white",
            arrows=True,
            arrowsize=30,
            linewidths=2,
            ax=ax,
            width=5.0,
        )

        node_cmap = (
            plt.cm.tab10(np.arange(10))
            if len(self.X_raw) <= 10
            else plt.cm.tab20(np.arange(20))
        )
        nx.draw(
            G,
            pos,
            node_size=[node["weight"] * 5000 for node in G.nodes.values()],
            node_color=node_color,
            edge_color=color_data,
            edgecolors="white",
            arrows=True,
            arrowsize=30,
            linewidths=2,
            cmap=ListedColormap(node_cmap[: len(self.X_raw)]),
            edge_cmap=plt.cm.Reds,
            ax=ax,
            alpha=1,
            width=5.0,
        )

        texts = []
        for node in G.nodes():
            text_ = ax.text(
                pos[node][0],
                pos[node][1],
                str(node),
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
            )
            text_.set_path_effects(
                [patheffects.withStroke(linewidth=3, foreground="w")]
            )
            texts.append(text_)

        if layout == "normal":
            adjust_text(texts)

        plt.show()

        if save:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

    def plot_fold_change(
        self,
        cluster_names,
        cluster1,
        cluster2,
        tf_gene_names=None,
        threshold=1.0,
        save=False,
        save_path=None,
    ):
        """
        Plot fold change between two clusters.

        Parameters
        ----------
        cluster_names : list of list of str
            1st dimension is the number of days, 2nd dimension is the number of gmm components
            in each day.
            Can be generaged by 'generate_cluster_names' method.

        cluster1 : str
            Cluster name of denominator.
            
        cluster2 : str
            Cluster name of numerator.

        tf_gene_names : list of str, optional
            List of transcription factor gene names to use, by default None
            If None, all gene names (self.gene_names) will be used.
            You can pass on any list of gene names you want to use, not limited to TF genes.

        threshold : float, optional
            Threshold to filter labels, by default 1.0
            Only genes with fold change greater than this threshold will be plotted its label.

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './fold_change.png'
        """
        
        if save and save_path is None:
            save_path = "./fold_change.png"

        if tf_gene_names is None:
            gene_names_to_use = self.gene_names
        else:
            gene_names_to_use = tf_gene_names

        genes = self.get_positive_gmm_mean_gene_values_per_cluster(
            self.get_gmm_means(),
            cluster_names=list(itertools.chain.from_iterable(cluster_names)),
        )
        genes = genes.loc[:, genes.columns.isin(gene_names_to_use)]
        genes = genes.T
        genes_fold_change = pd.DataFrame(index=genes.index)
        genes_fold_change[cluster1] = genes[cluster1]
        genes_fold_change[cluster2] = genes[cluster2]

        fig = go.Figure()
        gene_exp1 = genes_fold_change[cluster1]
        gene_exp2 = genes_fold_change[cluster2]
        fold_change_abs = (gene_exp2 - gene_exp1).abs()
        genes_fold_change = genes_fold_change.loc[
            genes_fold_change.index.isin(
                fold_change_abs[fold_change_abs.values > threshold].index
            ),
            :,
        ]

        fig.add_trace(
            go.Scatter(
                x=genes[cluster1],
                y=genes[cluster2],
                mode="markers",
                marker={
                    "symbol": "circle",
                    "color": "blue",
                    "opacity": 0.1,
                },
                text=genes.index,
                name=f"|FC| < {threshold}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=genes_fold_change[cluster1],
                y=genes_fold_change[cluster2],
                mode="markers+text",
                marker={
                    "symbol": "circle",
                    "color": "red",
                    "opacity": 0.8,
                },
                text=genes_fold_change.index,
                textposition="top center",
                name=f"|FC| > {threshold}",
            )
        )

        fig.update_layout(
            xaxis={"title": cluster1},
            yaxis={"title": cluster2},
            width=1000,
            height=800,
            showlegend=True,
            title=f"Fold Change (threshold = {threshold})",
        )
        fig.show()

        if save:
            fig.write_image(save_path)

    def plot_pathway_mean_var(
        self,
        cluster_names,
        pathway_names,
        tf_gene_names=None,
        threshold=1.0,
        save=False,
        save_path=None,
    ):
        """
        Plot mean and variance of gene expression levels within a pathway.

        Parameters
        ----------
        cluster_names : list of list of str
            1st dimension is the number of days, 2nd dimension is the number of gmm components
            in each day.
            Can be generaged by 'generate_cluster_names' method.
        
        pathway_names : list of str of shape (n_days,)
            List of cluster names included in the pathway.
            Specify like ['day0's cluster name', 'day1's cluster name', ..., 'dayN's cluster name'].
            
        tf_gene_names : list of str, optional
            List of transcription factor gene names to use, by default None
            If None, all gene names (self.gene_names) will be used.
            You can pass on any list of gene names you want to use, not limited to TF genes.

        threshold : float, optional
            Threshold to filter labels, by default 1.0
            Only genes with variance greater than this threshold will be plotted its label.

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './pathway_mean_var.png'
        """

        if save and save_path is None:
            save_path = "./pathway_mean_var.png"

        if tf_gene_names is None:
            gene_names_to_use = self.gene_names
        else:
            gene_names_to_use = tf_gene_names

        genes = self.get_positive_gmm_mean_gene_values_per_cluster(
            self.get_gmm_means(),
            cluster_names=list(itertools.chain.from_iterable(cluster_names)),
        )
        genes = genes.loc[:, genes.columns.isin(gene_names_to_use)]

        pathway_genes = genes.loc[pathway_names]
        mean = pathway_genes.mean(axis=0)
        var = pathway_genes.var(axis=0)
        pathway_mean_var = pd.DataFrame(
            mean, index=pathway_genes.columns, columns=["mean"]
        )
        pathway_mean_var["var"] = var

        fig = go.Figure()
        pathway_mean_var_above_thresh = pathway_mean_var.loc[
            pathway_mean_var.index.isin(
                pathway_mean_var[pathway_mean_var["var"].values > threshold].index
            ),
            :,
        ]
        text = pathway_mean_var.index
        text_above_thresh = pathway_mean_var_above_thresh.index

        fig.add_trace(
            go.Scatter(
                x=pathway_mean_var["mean"],
                y=pathway_mean_var["var"],
                mode="markers",
                marker={
                    "symbol": "circle",
                    "color": "blue",
                    "opacity": 0.1,
                },
                text=text,
                name=f"var < {threshold}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pathway_mean_var_above_thresh["mean"],
                y=pathway_mean_var_above_thresh["var"],
                mode="markers+text",
                marker={
                    "symbol": "circle",
                    "color": "red",
                    "opacity": 0.8,
                },
                text=text_above_thresh,
                textposition="top center",
                name=f"var > {threshold}",
            )
        )

        fig.update_layout(
            xaxis={"title": "mean"},
            yaxis={"title": "var"},
            width=1000,
            height=800,
            showlegend=True,
            title=f"pathway_mean_var (threshold = {threshold})",
        )
        fig.show()

        if save:
            fig.write_image(save_path)

    def plot_pathway_gene_expressions(
        self,
        cluster_names,
        pathway_names,
        selected_genes,
        save=False,
        save_path=None,
    ):
        """
        Plot gene expression levels within a pathway.

        Parameters
        ----------
        cluster_names : list of list of str
            1st dimension is the number of days, 2nd dimension is the number of gmm components
            in each day.
            Can be generaged by 'generate_cluster_names' method.

        pathway_names : list of str of shape (n_days,)
            List of cluster names included in the pathway.
            Specify like ['day0's cluster name', 'day1's cluster name', ..., 'dayN's cluster name'].

        selected_genes : list of str
            List of gene names whose gene expression changes you want to track.
            Recommend using about 5 genes.

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './pathway_gene_expressions.png'
        """

        if save and save_path is None:
            save_path = "./pathway_gene_expressions.png"

        genes = self.get_positive_gmm_mean_gene_values_per_cluster(
            self.get_gmm_means(),
            cluster_names=list(itertools.chain.from_iterable(cluster_names)),
        )

        pathway_selected_genes = genes.loc[pathway_names].loc[:, selected_genes]
        fig = go.Figure()
        for gene_name in pathway_selected_genes.columns:
            fig.add_trace(
                go.Scatter(
                    x=pathway_selected_genes.index,
                    y=pathway_selected_genes[gene_name].values,
                    mode="lines+markers",
                    name=gene_name,
                    line_shape="spline",
                )
            )

        fig.update_layout(
            xaxis={"title": "day"},
            yaxis={"title": "loag(gene_exp+1)"},
            width=1200,
            height=800,
            showlegend=True,
            title="pathway_gene_expressions",
        )
        fig.show()

        if save:
            fig.write_image(save_path)

    def plot_pathway_single_gene_2d(
        self, gene_name, mode="pca", col=None, save=False, save_path=None
    ):
        warnings.warn(
            "'plot_pathway_single_gene_2d()' was deprecated and will be removed in future versions.\n"
            "Use 'plot_gene_expression_2d()' instead.",
            FutureWarning,
        )
        self.plot_gene_expression_2d(gene_name, mode, col, save, save_path)

    def plot_gene_expression_2d(
        self, gene_name, mode="pca", col=None, save=False, save_path=None
    ):
        """
        Plot gene expression levels in 2D space.

        Parameters
        ----------
        gene_name : str
            Gene name to plot expression level.

        mode : {'pca', 'umap'}, optional
            The space to plot gene expression levels, by default "pca"

        col : list or tuple of str of shape (2,), optional
            X and Y axis labels, by default None
            If None, the first two columns of the input data will be used.

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './pathway_single_gene_2d.png'

        Raises
        ------
        ValueError
            When 'mode' is not 'pca' or 'umap'.
        """
        
        if mode not in ["pca", "umap"]:
            raise ValueError("The parameter 'mode' should be 'pca' or 'umap'.")

        if save and save_path is None:
            save_path = "./pathway_single_gene_2d.png"

        X_concated = pd.concat(self.X_pca if mode == "pca" else self.X_umap)
        if col:
            x_col, y_col = col
        else:
            x_col, y_col = X_concated.columns[0], X_concated.columns[1]
        fig = px.scatter(
            X_concated,
            x=x_col,
            y=y_col,
            color=pd.concat(self.X_normalized)[gene_name],
            hover_name=X_concated.index,
            width=1000,
            height=800,
        )
        fig.update_layout(title=f"gene = {gene_name}")
        fig.show()

        if save:
            fig.write_image(save_path)

    def plot_pathway_single_gene_3d(
        self, gene_name, col=None, save=False, save_path=None
    ):
        warnings.warn(
            "'plot_pathway_single_gene_3d()' was deprecated and will be removed in future versions.\n"
            "Use 'plot_gene_expression_3d()' instead.",
            FutureWarning,
        )
        self.plot_gene_expression_3d(gene_name, col, save, save_path)

    def plot_gene_expression_3d(self, gene_name, col=None, save=False, save_path=None):
        """
        Plot gene expression levels in 3D space.

        Parameters
        ----------
        gene_name : str
            Gene name to plot expression level.

        col : list or tuple of str of shape (2,), optional
            X, Y, and Z axis labels, by default None
            If None, the first three columns of the input data will be used.
            
        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './pathway_single_gene_3d.html'
        """

        if save and save_path is None:
            save_path = "./pathway_single_gene_3d.html"

        X_concated = pd.concat(self.X_pca)
        if col:
            x_col, y_col, z_col = col
        else:
            x_col, y_col, z_col = (
                X_concated.columns[0],
                X_concated.columns[1],
                X_concated.columns[2],
            )
        fig = px.scatter_3d(
            X_concated,
            x=x_col,
            y=y_col,
            z=z_col,
            color=pd.concat(self.X_normalized)[gene_name],
            hover_name=X_concated.index,
            width=1000,
            height=800,
        )
        fig.update_traces(marker_size=1)
        fig.update_layout(title=f"gene = {gene_name}")
        fig.show()

        if save:
            fig.write_html(save_path)

    def make_interpolation_data(
        self, gmm_source, gmm_target, t, columns=None, n_samples=2000, seed=0
    ):
        """
        Make interpolation data between two timepoints.

        Parameters
        ----------
        gmm_source : GaussianMixture
            GMM model of the source timepoint.

        gmm_target : GaussianMixture
            GMM model of the target timepoint.

        t : float
            Interpolation ratio.
            0 <= t <= 1.
            0 is the source timepoint, 1 is the target timepoint.
            If you specify 0.5, the data will be interpolated halfway between the source and target timepoints.

        columns : list of str, optional
            Columns names of the output data, by default None

        n_samples : int, optional
            Number of samples to generate, by default 2000

        seed : int, optional
            Random seed, by default 0

        Returns
        -------
        pd.DataFrame
            Interpolated data between two timepoints.
        """
        
        d = gmm_source.means_.shape[1]
        K_0, K_1 = gmm_source.means_.shape[0], gmm_target.means_.shape[0]

        mu_0, mu_1 = gmm_source.means_, gmm_target.means_
        S_0, S_1 = gmm_source.covariances_, gmm_target.covariances_

        pi_0, pi_1 = gmm_source.weights_, gmm_target.weights_

        solution = self.egot(
            np.ravel(pi_0),
            np.ravel(pi_1),
            mu_0.reshape(K_0, d),
            mu_1.reshape(K_1, d),
            S_0.reshape(K_0, d, d),
            S_1.reshape(K_1, d, d),
        )
        pit = solution.reshape(K_0 * K_1, 1).T

        mut, St = self.calculate_mut_st(gmm_source, gmm_target, t)

        K = mut.shape[0]
        pit = pit.reshape(1, K)
        means = mut
        covariances = St
        weights = pit[0, :]
        rng = check_random_state(seed)
        n_samples_comp = rng.multinomial(n_samples, weights)
        X_interpolation = np.vstack(
            [
                rng.multivariate_normal(mean, covariance, int(sample))
                for (mean, covariance, sample) in zip(
                    means, covariances, n_samples_comp
                )
            ]
        )
        X_interpolation = pd.DataFrame(X_interpolation, columns=columns)
        return X_interpolation

    def plot_true_and_interpolation_distributions(
        self,
        interpolate_index,
        mode="pca",
        n_samples=2000,
        t=0.5,
        plot_source_and_target=True,
        alpha_true=0.5,
        x_col_name=None,
        y_col_name=None,
        x_range=None,
        y_range=None,
        save=False,
        save_path=None,
    ):
        """
        Compare the true and interpolation distributions by plotting them.

        Parameters
        ----------
        interpolate_index : int
            Index of the timepoint to interpolate.
            1 <= interpolate_index <= n_days - 2

        mode : {'pca', 'umap'}, optional
            The space to plot gene expression levels, by default "pca"

        n_samples : int, optional
            Number of samples to generate, by default 2000

        t : float, optional
            Interpolation ratio, by default 0.5
            If you want to interpolate halfway between the source and target timepoints, specify 0.5.
            Source timepoint is interpolate_index - 1, target timepoint is interpolate_index + 1.

        plot_source_and_target : bool, optional
            If True, plot the source and target timepoints, by default True

        alpha_true : float, optional
            Transparency of the true data, by default 0.5

        x_col_name : str, optional
            Label of the x-axis, by default None

        y_col_name : str, optional
            Label of the y-axis, by default None

        x_range : list or tuple of float of shape (2,), optional
            Range of the x-axis, by default None
            If None, the range will be automatically determined based on the data.

        y_range : list or tuple of float of shape (2,), optional
            Range of the y-axis, by default None
            If None, the range will be automatically determined based on the data.

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None

        Raises
        ------
        ValueError
            When 'mode' is not 'pca' or 'umap'.
        """
        
        if mode not in ["pca", "umap"]:
            raise ValueError("The parameter 'mode' should be 'pca' or 'umap'.")

        X_interpolation = self.make_interpolation_data(
            self.gmm_models[interpolate_index - 1],
            self.gmm_models[interpolate_index + 1],
            t,
            self.X_pca[0].columns,
            n_samples=n_samples,
        )
        if mode == "umap":
            if self.verbose:
                print("Transforming interpolated data with UMAP...")
            X_interpolation = pd.DataFrame(
                self.umap_model.transform(X_interpolation),
                columns=self.X_umap[0].columns,
            )
        X_true = (
            self.X_pca[interpolate_index]
            if mode == "pca"
            else self.X_umap[interpolate_index]
        )
        X_source = (
            self.X_pca[interpolate_index - 1]
            if mode == "pca"
            else self.X_umap[interpolate_index - 1]
        )
        X_target = (
            self.X_pca[interpolate_index + 1]
            if mode == "pca"
            else self.X_umap[interpolate_index + 1]
        )

        if x_range is None or y_range is None:
            df_concated = pd.concat([X_source, X_target, X_true, X_interpolation])
            if x_range is None:
                x_range = (
                    np.min(df_concated.iloc[:, 0]) - 5,
                    np.max(df_concated.iloc[:, 0]) + 5,
                )
            if y_range is None:
                y_range = (
                    np.min(df_concated.iloc[:, 1]) - 5,
                    np.max(df_concated.iloc[:, 1]) + 5,
                )
        if x_col_name is None:
            x_col_name = X_true.columns[0]
        if y_col_name is None:
            y_col_name = X_true.columns[1]

        if save and save_path is None:
            save_path = "./true_and_interpolation_distributions.png"

        plt.scatter(
            X_interpolation[x_col_name].values,
            X_interpolation[y_col_name].values,
            marker="o",
            color="orange",
            label="interpolation",
            alpha=0.5,
        )
        plt.scatter(
            X_true[x_col_name].values,
            X_true[y_col_name].values,
            marker="o",
            color="red",
            label="true",
            alpha=alpha_true,
        )
        if plot_source_and_target:
            plt.scatter(
                X_source[x_col_name].values,
                X_source[y_col_name].values,
                marker="o",
                color="blue",
                label="source",
                alpha=0.5,
            )
            plt.scatter(
                X_target[x_col_name].values,
                X_target[y_col_name].values,
                marker="o",
                color="green",
                label="target",
                alpha=0.5,
            )
        plt.xlabel(x_col_name)
        plt.ylabel(y_col_name)
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.legend(loc=0)
        plt.title("true and interpolation distributions")

        if save:
            plt.savefig(save_path)

        plt.show()

    def animate_gene_expression(
        self,
        target_gene_name,
        mode="pca",
        interpolate_interval=11,
        n_samples=5000,
        x_range=None,
        y_range=None,
        c_range=None,
        x_label=None,
        y_label=None,
        cmap="gnuplot2",
        save=False,
        save_path=None,
    ):
        """
        Calculate interpolation between all timepoints and create animation colored by gene expression level. 

        Parameters
        ----------
        target_gene_name : str
            Gene name to plot expression level.
        mode : {'pca', 'umap'}, optional
            The space to plot gene expression levels, by default "pca"
        interpolate_interval : int, optional
            Number of frames to interpolate between two timepoints, by default 11
            This is the total number of frames at both timepoints and the number of frames
            between these.
            Note that both ends are included.
            
        n_samples : int, optional
            Number of samples to generate, by default 5000

        x_range : list or tuple of float of shape (2,), optional
            Range of the x-axis, by default None

        y_range : list or tuple of float of shape (2,), optional
            Range of the y-axis, by default None

        c_range : list or tuple of float of shape (2,), optional
            Range of the color bar, by default None

        x_label : str, optional
            Label of the x-axis, by default None

        y_label : str, optional
            Label of the y-axis, by default None

        cmap : str, optional
            String of the colormap, by default "gnuplot2"

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './interpolate_video.gif'

        Raises
        ------
        ValueError
            When 'mode' is not 'pca' or 'umap'.
        """

        if mode not in ["pca", "umap"]:
            raise ValueError("The parameter 'mode' should be 'pca' or 'umap'.")

        X = self.X_pca if mode == "pca" else self.X_umap
        gene_expression_level = pd.concat(self.X_selected)[target_gene_name]

        if x_range is None or y_range is None:
            X_concated = pd.concat(X)
            if x_range is None:
                x_range = (
                    np.min(X_concated.iloc[:, 0]) - 5,
                    np.max(X_concated.iloc[:, 0]) + 5,
                )
            if y_range is None:
                y_range = (
                    np.min(X_concated.iloc[:, 1]) - 5,
                    np.max(X_concated.iloc[:, 1]) + 5,
                )
        if c_range is None:
            c_range = (
                np.min(gene_expression_level),
                np.max(gene_expression_level),
            )

        if save and save_path is None:
            save_path = "./interpolate_video.gif"

        fig, ax = plt.subplots(figsize=(10, 8))
        ims = []
        for i in range(len(self.gmm_models) - 1):
            t = np.linspace(0, 1, interpolate_interval)
            if self.verbose:
                print(
                    f"Interpolating between {self.day_names[i]} and {self.day_names[i + 1]}..."
                )
            frames = range(int(i > 0), interpolate_interval)
            for j in tqdm(frames) if self.verbose else frames:
                if i != 0 and j == 0:
                    continue
                X_interpolation = self.make_interpolation_data(
                    self.gmm_models[i],
                    self.gmm_models[i + 1],
                    t[j],
                    columns=self.X_pca[0].columns,
                    n_samples=n_samples,
                )
                X_genes_interpolation = self.pca_model.inverse_transform(
                    X_interpolation.values
                )
                df_genes_interpolation = pd.DataFrame(
                    X_genes_interpolation, columns=self.gene_names
                )

                if mode == "umap":
                    X_interpolation = pd.DataFrame(
                        self.umap_model.transform(X_interpolation),
                        columns=self.X_umap[0].columns,
                    )

                im = plt.scatter(
                    X_interpolation.iloc[:, [0]],
                    X_interpolation.iloc[:, [1]],
                    c=df_genes_interpolation[target_gene_name].values,
                    s=0.5,
                    alpha=0.8,
                    cmap=cmap,
                )
                plt.xlabel(x_label or X_interpolation.columns[0])
                plt.ylabel(y_label or X_interpolation.columns[1])
                plt.xlim(x_range)
                plt.ylim(y_range)
                plt.clim(c_range)
                if i == 0 and j == 0:
                    plt.colorbar()
                title = ax.text(
                    0.5,
                    1.0,
                    f"{self.day_names[i]}, t = {t[j]:.1f}, gene = {target_gene_name}",
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=20,
                )
                ims.append([im] + [title])

        if self.verbose:
            print("Creating animation...")
        anim_gene = animation.ArtistAnimation(fig, ims, interval=100, repeat=False)
        if is_notebook():
            display(HTML(anim_gene.to_jshtml()))
        else:
            plt.show()

        if save:
            anim_gene.save(save_path, writer="pillow")

        plt.close()

    def get_gaussian_map(self, m_0, m_1, sigma_0, sigma_1, x):
        d = sigma_0.shape[0]
        m_0 = m_0.reshape(1, d)
        m_1 = m_1.reshape(1, d)
        sigma_0 = sigma_0.reshape(d, d)
        sigma_1 = sigma_1.reshape(d, d)
        sigma = np.linalg.pinv(sigma_0) @ spl.sqrtm(sigma_0 @ sigma_1)
        Tx = m_1 + (x - m_0) @ sigma
        return Tx

    def _calculate_cell_velocity(self, gmm_source, gmm_target, X_item, solution):
        d = gmm_source.means_.shape[1]
        K_0, K_1 = gmm_source.means_.shape[0], gmm_target.means_.shape[0]

        mu_0, mu_1 = gmm_source.means_, gmm_target.means_
        S_0, S_1 = gmm_source.covariances_, gmm_target.covariances_

        n = X_item.shape[0]
        T = np.zeros((K_0, K_1, d, n))
        barycentric_projection_map = np.zeros((d, n))
        Nj = np.zeros((n, K_0))

        for i in range(K_0):
            for j in range(K_1):
                T[i, j, :, :] = self.get_gaussian_map(
                    mu_0[i, :], mu_1[j, :], S_0[i, :, :], S_1[j, :, :], X_item.values
                ).T
        for i in range(K_0):
            logprob = gmm_source.score_samples(X_item.values)
            with np.errstate(divide="ignore"):
                Nj[:, i] = np.exp(
                    np.log(
                        multivariate_normal.pdf(
                            X_item.values, mean=mu_0[i, :], cov=S_0[i, :, :]
                        )
                    )
                    - logprob
                )
        for i in range(K_0):
            for j in range(K_1):
                barycentric_projection_map += (
                    solution[i, j] * Nj[:, i].T * T[i, j, :, :]
                )
        velo = barycentric_projection_map.T - X_item.values
        return velo

    def calculate_cell_velocities(self):
        """
        Calculate cell velocities between each day.

        Returns
        -------
        pd.DataFrame
            Cell velocities between each day.
            The rows are ordered as follows:
            when the number of days is N and the number of cells in each day is M_1, M_2, ..., M_N,
            [day1_cell1 -> day1_cell2 -> ... -> day1_cellM_1 -> day2cell1 -> ... -> day(N-1)cellM_N]
        """

        velocities = pd.DataFrame(
            columns=self.X_pca[0].columns
        )

        if self.solutions is None:
            self.solutions = self.calculate_solutions(self.gmm_models)

        if self.verbose:
            print("Calculating cell velocities between each day...")
        for i in (
            tqdm(range(len(self.gmm_models) - 1))
            if self.verbose
            else range(len(self.gmm_models) - 1)
        ):
            gmm_source = self.gmm_models[i]
            gmm_target = self.gmm_models[i + 1]

            velocity = self._calculate_cell_velocity(
                gmm_source, gmm_target, self.X_pca[i], self.solutions[i]
            )

            velocity = pd.DataFrame(
                velocity,
                columns=(
                    self.X_pca[0].columns
                ),
            )
            velocities = pd.concat([velocities, velocity])

        return velocities

    def plot_cell_velocity(
        self,
        velocities,
        mode="pca",
        color_points="gmm",
        size_points=30,
        cmap="tab20",
        cluster_names=None,
        save=False,
        save_path=None,
    ):
        """
        Plot cell velocities in 2D space.

        Parameters
        ----------
        velocities : pd.DataFrame
            Cell velocities calculated by 'calculate_cell_velocities' method.

        mode : {'pca' or 'umap'}, optional
            The space to plot cell velocities, by default "pca"

        color_points : {'gmm' or 'day'}, optional
            Color points by GMM clusters or days, by default "gmm"

        size_points : int, optional
            Size of points, by default 30

        cmap : str, optional
            String of matplolib colormap name, by default "tab20"

        cluster_names : list of str of shape (sum of gmm components), optional
            List of gmm cluster names, by default None
            Used when 'color_points' is 'gmm'.
            You need to flatten the list of lists of gmm cluster names before passing it.

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './cell_velocity.png'

        Raises
        ------
        ValueError
            This error is raised in the following cases:
            - When 'mode' is not 'pca' or 'umap'.
            - When 'color_points' is not 'gmm' or 'day'.
            - When 'color_points' is 'gmm' and 'cluster_names' is None.
        """

        if mode not in ["pca", "umap"]:
            raise ValueError("The parameter 'mode' should be 'pca' or 'umap'.")

        if color_points not in ["gmm", "day"]:
            raise ValueError(
                "The parameter 'color_points' should be None, 'gmm', or 'day'."
            )
            
        if color_points == "gmm" and cluster_names is None:
            raise ValueError("The parameter 'cluster_names' should be specified when 'color_points' is 'gmm'.")
            
        if save and save_path is None:
            save_path = "./cell_velocity.png"

        day_labels = []
        for i in range(len(self.day_names) - 1):
            day_labels += [i for _ in range(len(self.X_pca[i]))]

        adata_cvel = anndata.AnnData(
            pd.concat(self.X_pca[:-1]).values,
            obs=pd.DataFrame(day_labels, index=pd.concat(self.X_pca[:-1]).index, columns=["clusters"]),
            obsm={"X_pca": pd.concat(self.X_pca[:-1]).values},
            layers={
                "velocity": velocities.values,
                "spliced": pd.concat(self.X_pca[:-1]).values,
            },
        )
        if mode == "umap":
            adata_cvel.obsm["X_umap"] = pd.concat(self.X_umap[:-1]).values
            
        neighbors(adata_cvel, use_rep="X_pca", n_neighbors=self.pca_model.n_components_)
        scv.tl.velocity_graph(adata_cvel)
        
        figsize = (8, 6)
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        scv.pl.velocity_embedding_stream(
            adata_cvel,
            basis=mode,
            color="black",
            vkey="velocity",
            title="",
            density=2,
            alpha=0.0,
            fontsize=14,
            legend_fontsize=0,
            legend_loc=None,
            arrow_size=1,
            linewidth=1.5,
            ax=ax,
            show=False,
            X_grid=None,
            V_grid=None,
            sort_order=True,
            size=50,
            colorbar=False,
        )

        X = self.X_pca if mode == "pca" else self.X_umap
        colors = []
        if color_points == "gmm":
            label_sum = 0
            for i in range(len(self.gmm_labels)):
                colors += [label + label_sum for label in self.gmm_labels_modified[i]]
                label_sum += self.gmm_n_components_list[i]
        elif color_points == "day":
            for i in range(len(X)):
                colors += [i] * len(X[i])

        scatter = plt.scatter(
            pd.concat(X).iloc[:, 0],
            pd.concat(X).iloc[:, 1],
            cmap=plt.get_cmap(cmap, len(set(colors))),
            c=colors,
            edgecolors="w",
            linewidth=0.5,
            s=size_points,
            alpha=0.5,
        )
        if color_points == "gmm" and cluster_names is not None:
            handles = scatter.legend_elements(num=list(range(len(cluster_names))))[
                0
            ]
            labels = cluster_names
        else:
            handles = scatter.legend_elements(num=list(range(len(self.day_names))))[
                0
            ]
            labels = self.day_names
        plt.legend(
            handles=handles,
            labels=labels,
        )

        ax.axis("off")

        plt.show()

        if save:
            fig.savefig(save_path)

    def plot_interpolation_of_cell_velocity(
        self,
        velocities,
        mode="pca",
        color_streams=False,
        color_points="gmm",
        cluster_names=None,
        x_range=None,
        y_range=None,
        cmap="gnuplot2",
        linspace_num=300,
        save=False,
        save_path=None,
    ):
        """
        .. warning::
            ``plot_interpolation_of_cell_velocity()`` was deprecated in version 0.3.0 and will be removed in future versions.
            Use ``plot_cell_velocity()`` instead.
        
        Parameters
        ----------
        velocities : pd.DataFrame
            Cell velocities calculated by 'calculate_cell_velocities' method.

        mode : {'pca', 'umap'}, optional
            The space to plot cell velocities, by default "pca"

        color_streams : bool, optional
            If True, color the streamlines by the speed of the cell velocities, by default False

        color_points : {'gmm' or 'day'}, optional
            Color points by GMM clusters or days, by default "gmm"

        cluster_names : list of str of shape (sum of gmm n_components), optional
            List of gmm cluster names, by default None
            Used when 'color_points' is 'gmm'.
            You need to flatten the list of lists of gmm cluster names before passing it.   

        x_range : tuple or list of float of shape (2,), optional
            Limit of the x-axis, by default None

        y_range : tuple or list of float of shape (2,), optional
            Limit of the y-axis, by default None

        cmap : str, optional
            String of matplolib colormap name, by default "gnuplot2"

        linspace_num : int, optional
            Number of points on each axis to interpolate, by default 300
            linspace_num * linspace_num points will be interpolated.

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './interpolation_of_cell_velocity_gmm_clusters.png'
            
        Raises
        ------
        ValueError
            This error is raised in the following cases:
            - When 'mode' is not 'pca' or 'umap'.
            - When 'color_points' is not 'gmm' or 'day'.
            - When 'color_points' is 'gmm' and 'cluster_names' is None.
        """

        warnings.warn(
            "'plot_interpolation_of_cell_velocity()' was deprecated and will be removed in future versions.\n"
            "Use 'plot_cell_velocity()' instead.",
            FutureWarning,
        )

        if mode not in ["pca", "umap"]:
            raise ValueError("The parameter 'mode' should be 'pca' or 'umap'.")
        if color_points is not None and color_points not in ["gmm", "day"]:
            raise ValueError(
                "The parameter 'color_points' should be None, 'gmm', or 'day'."
            )
        if color_points == "gmm" and cluster_names is None:
            raise ValueError("The parameter 'cluster_names' should be specified when 'color_points' is 'gmm'.")
        
        if save and save_path is None:
            save_path = "./interpolation_of_cell_velocity_gmm_clusters.png"

        X = self.X_pca if mode == "pca" else self.X_umap

        colors = []
        if color_points == "gmm":
            label_sum = 0
            for i in range(len(self.gmm_labels)):
                colors += [label + label_sum for label in self.gmm_labels_modified[i]]
                label_sum += self.gmm_n_components_list[i]
        elif color_points == "day":
            for i in range(len(X)):
                colors += [i] * len(X[i])

        X_concated = pd.concat(X)
        x, y = np.meshgrid(
            np.linspace(
                np.min(X_concated.iloc[:, 0]) - 5,
                np.max(X_concated.iloc[:, 0]) + 5,
                linspace_num,
            ),
            np.linspace(
                np.min(X_concated.iloc[:, 1]) - 5,
                np.max(X_concated.iloc[:, 1]) + 5,
                linspace_num,
            ),
        )
        
        if mode == "umap":
            velocities = self.umap_model.transform(velocities.values + pd.concat(self.X_pca[:-1]).values) - pd.concat(self.X_umap[:-1]).values
            x_velocity = velocities[:, 0]
            y_velocity = velocities[:, 1]
        else:
            x_velocity = velocities.iloc[:, 0]
            y_velocity = velocities.iloc[:, 1]

        points = np.transpose(
            np.vstack((pd.concat(X[:-1]).iloc[:, 0], pd.concat(X[:-1]).iloc[:, 1]))
        )
        stream_x_velocity = interpolate.griddata(
            points, x_velocity, (x, y), method="linear", fill_value=0
        )
        stream_y_velocity = interpolate.griddata(
            points, y_velocity, (x, y), method="linear", fill_value=0
        )
        stream_speed = np.sqrt(stream_x_velocity**2 + stream_y_velocity**2)

        plt.figure(figsize=(10, 8))
        stream = plt.streamplot(
            x,
            y,
            stream_x_velocity,
            stream_y_velocity,
            color=stream_speed if color_streams else "black",
            cmap=cmap,
            density=3.0,
        )
        if color_points in ["gmm", "day"]:
            scatter = plt.scatter(
                pd.concat(X).iloc[:, 0],
                pd.concat(X).iloc[:, 1],
                c=colors,
                cmap=plt.get_cmap(cmap, len(set(colors))),
                s=20,
                alpha=0.5,
            )
            if color_points == "gmm" and cluster_names is not None:
                handles = scatter.legend_elements(num=list(range(len(cluster_names))))[
                    0
                ]
                labels = cluster_names
            else:
                handles = scatter.legend_elements(num=list(range(len(self.day_names))))[
                    0
                ]
                labels = self.day_names
            plt.legend(
                handles=handles,
                labels=labels,
            )
        else:
            plt.scatter(
                pd.concat(X).iloc[:, 0],
                pd.concat(X).iloc[:, 1],
                c="gray",
                s=20,
                alpha=0.3,
            )

        plt.xlabel(X[0].columns[0])
        plt.ylabel(X[0].columns[1])

        if x_range is not None:
            plt.xlim(x_range)
        if y_range is not None:
            plt.ylim(y_range)

        if color_streams:
            plt.colorbar(stream.lines)

        plt.show()

        if save:
            plt.savefig(save_path)

    def calculate_grns(
        self,
        selected_clusters=None,
        alpha_range=(-2, 2),
        cv=3,
        ridge_cv_fit_intercept=False,
        ridge_fit_intercept=False,
    ):
        """
        Calculate gene regulatory networks (GRNs) between each day.

        Parameters
        ----------
        selected_clusters : list of list of int of shape (n_days, 2), optional
            Specify the clusters to calculate GRNs, by default None
            If None, all clusters will be used.
            The list should be like 
            [[day1's index, selected cluster number], [day2's index, selected cluster number], ...].

        alpha_range : tuple or list of float of shape (2,), optional
            Range of alpha values for Ridge regression, by default (-2, 2)

        cv : int, optional
            Number of cross-validation folds, by default 3
            This parameter is passed to RidgeCV's 'cv' parameter.

        ridge_cv_fit_intercept : bool, optional
            Whether to calculate the intercept in RidgeCV, by default False
            This parameter is passed to RidgeCV's 'fit_intercept' parameter.

        ridge_fit_intercept : bool, optional
            Whether to calculate the intercept in Ridge, by default False
            This parameter is passed to Ridge's 'fit_intercept' parameter.

        Returns
        -------
        list of pd.DataFrame
            Gene regulatory networks between each day.
            The rows and columns are gene names.
            Each element of the list corresponds to the GRN between day i and day i + 1.
        
        list of RidgeCV objects
            RidgeCV objects used to calculate GRNs.
            Each element of the list corresponds to the RidgeCV object between day i and day i + 1.
        """

        grns, ridge_cvs = [], []

        if self.solutions is None:
            self.solutions = self.calculate_solutions(self.gmm_models)

        if self.verbose:
            print("Calculating GRNs between each day...")
        for i in (
            tqdm(range(len(self.gmm_models) - 1))
            if self.verbose
            else range(len(self.gmm_models) - 1)
        ):
            gmm_source = self.gmm_models[i]
            gmm_target = self.gmm_models[i + 1]

            velo = self._calculate_cell_velocity(
                gmm_source, gmm_target, self.X_pca[i], self.solutions[i]
            )

            if selected_clusters is None:
                X_, V_ = self.X_pca[i], velo
            else:
                X_ = self.X_pca[selected_clusters[i][0]][
                    self.gmm_labels_modified[selected_clusters[i][0]]
                    == selected_clusters[i][1]
                ]

                V_ = velo[
                    self.gmm_labels_modified[selected_clusters[i][0]]
                    == selected_clusters[i][1]
                ]

            alphas_cv = np.logspace(alpha_range[0], alpha_range[1], num=20)
            ridge_cv = linear_model.RidgeCV(
                alphas=alphas_cv, cv=cv, fit_intercept=ridge_cv_fit_intercept
            )
            ridge_cv.fit(X_, V_)
            ridge_cvs.append(ridge_cv)

            grn = linear_model.Ridge(
                alpha=ridge_cv.alpha_, fit_intercept=ridge_fit_intercept
            )
            grn.fit(X_, V_)
            grn = pd.DataFrame(
                self.pca_model.components_.T @ grn.coef_ @ self.pca_model.components_,
                index=self.gene_names,
                columns=self.gene_names,
            )
            grns.append(grn)

        return grns, ridge_cvs

    def _make_grn_graph(self, df, threshold=0.01):
        graph = pydotplus.Dot(graph_type="digraph")
        for c in df.columns:
            node = pydotplus.Node(f'"{c}"', label=c)
            graph.add_node(node)
        for i in df.index:
            for c in df.columns:
                val = df.loc[i, c]
                if abs(val) > threshold:
                    edge = pydotplus.Edge(
                        graph.get_node(f'"{i}"')[0], graph.get_node(f'"{c}"')[0]
                    )
                    edge.set_label("{:.2f}".format(df.loc[i, c]))
                    edge.set_penwidth(
                        20 * (abs(val) - threshold) / (1 - threshold) + 0.5
                    )
                    h, s, v = 0.0 if val > 0 else 2 / 3, abs(val) + 1, 1.0
                    edge.set_color(" ".join([str(i) for i in (h, s, v)]))
                    graph.add_edge(edge)
        return graph

    def plot_grn_graph(
        self,
        grns,
        ridge_cvs,
        selected_genes,
        threshold=0.01,
        save=False,
        save_paths=None,
        save_format="png",
    ):
        """
        Plot gene regulatory networks (GRNs) between each day.

        Parameters
        ----------
        grns : list of pd.DataFrame
            Gene regulatory networks between each day.
            The rows and columns are gene names.

        ridge_cvs : list of RidgeCV objects
            RidgeCV objects used to calculate GRNs.

        selected_genes : list of str
            Gene names to plot GRNs.

        threshold : float, optional
            Threshold to plot edges, by default 0.01
            If the absolute value of the edge weight is less than this value, the edge will not be plotted.

        save : bool, optional
            If True, save the output image, by default False

        save_paths : str, optional
            Paths to save the output images, by default None

        save_format : str, optional
            Format of the output images, by default "png"
        """
        
        if save and save_paths is None:
            save_paths = [f"./grn_graph_{i + 1}.png" for i in range(len(grns))]
        for i, grn in enumerate(grns):
            if self.verbose:
                print(f"alpha = {ridge_cvs[i].alpha_}")
            grn_graph = self._make_grn_graph(
                grn[selected_genes].loc[selected_genes], threshold=threshold
            )
            if is_notebook():
                display(Image(grn_graph.create(format=save_format)))
            else:
                img = PILImage.open(BytesIO(grn_graph.create(format=save_format)))
                img.show()
            if save:
                grn_graph.write(save_paths[i], format=save_format)

    def calculate_waddington_potential(
        self,
        n_neighbors=100,
        knn_other_params={},
    ):
        """
        Calculate Waddington potential of each sample.
        
        Parameters
        ----------
        n_neighbors : int, optional
            Number of neighbors for rach sample, by default 100
            This parameter is passed to 'kneighbors_graph' function.
            
        knn_other_params : dict, optional
            Other parameters for 'kneighbors_graph' function, by default {}

        Returns
        -------
        np.ndarray of shape (sum of n_samples of each day - n_samples of the last day,)
            Waddington potential of each sample.
        """
        
        if self.solutions is None:
            self.solutions = self.calculate_solutions(self.gmm_models)

        F_all = []

        if self.verbose:
            print("Calculating F between each day...")
        for i in (
            tqdm(range(len(self.X_pca) - 1))
            if self.verbose
            else range(len(self.X_pca) - 1)
        ):
            solution = self.solutions[i]

            gmm_source, gmm_target = self.gmm_models[i], self.gmm_models[i + 1]

            d = gmm_source.means_.shape[1]
            K_0, K_1 = gmm_source.means_.shape[0], gmm_target.means_.shape[0]

            mu_0, mu_1 = gmm_source.means_, gmm_target.means_
            S_0, S_1 = gmm_source.covariances_, gmm_target.covariances_

            pi_0 = gmm_source.weights_

            B = 0
            F = 0
            for j in range(K_0):
                B = B + np.nan_to_num(
                    np.dot(
                        np.linalg.pinv(S_0)[j, :, :],
                        (self.X_pca[i].values - mu_0[j, :]).T,
                    )
                    * pi_0[j]
                    * multivariate_normal.pdf(
                        self.X_pca[i].values, mean=mu_0[j, :], cov=S_0[j, :, :]
                    )
                    / self.gaussian_mixture_density(
                        mu_0, S_0, pi_0, self.X_pca[i].values
                    )
                )
            for j in range(K_0):
                A = np.dot(
                    np.linalg.pinv(S_0)[j, :, :], (self.X_pca[i].values - mu_0[j, :]).T
                )
                I_1 = -A + B
                for k in range(K_1):
                    P = (
                        solution[j, k]
                        * multivariate_normal.pdf(
                            self.X_pca[i].values, mean=mu_0[j, :], cov=S_0[j, :, :]
                        )
                        / self.gaussian_mixture_density(
                            mu_0, S_0, pi_0, self.X_pca[i].values
                        ).T
                    )
                    P = np.nan_to_num(P)
                    Tmap = self.get_gaussian_map(
                        mu_0[j, :],
                        mu_1[k, :],
                        S_0[j, :, :],
                        S_1[k, :, :],
                        self.X_pca[i].values,
                    ).T
                    Tmap = Tmap.real
                    I_2 = np.nan_to_num(
                        np.trace(
                            np.linalg.pinv(S_0[j, :, :])
                            @ spl.sqrtm(S_0[j, :, :] @ S_1[k, :, :])
                        )
                    )
                    I_2 = I_2.real
                    F = F + np.sum(np.dot(I_1, Tmap.T)) * P + I_2 * P
            F = F - d
            F_all = np.append(F_all, F)

        if self.verbose:
            print("Applying knn ...")
        knn = kneighbors_graph(
            X=pd.concat(self.X_pca[:-1]).iloc[:, :2].values,
            n_neighbors=n_neighbors,
            mode="distance",
            metric="euclidean",
            **knn_other_params,
        )

        if self.verbose:
            print("Computing kernel ...")
        sim = lil_matrix(knn.shape)

        nonzero = knn.nonzero()
        sig = 10
        sim[nonzero] = np.exp(-np.array(knn[nonzero]) ** 2 / sig**2)
        sim = (sim + sim.T) / 2
        deg = sim.sum(axis=1)
        n = pd.concat(self.X_pca[:-1]).iloc[:, :2].values.shape[0]
        dia = np.diag(
            np.array(deg).reshape(
                n,
            )
        )
        lap = dia - sim
        lap = csc_matrix(np.array(lap))
        waddington_potential, *_ = spl_sparse.lsqr(lap, F_all)

        waddington_potential = zscore(waddington_potential)

        return waddington_potential, F_all

    def plot_waddington_potential(
        self,
        waddington_potential,
        mode="pca",
        gene_name=None,
        save=False,
        save_path=None,
    ):
        """
        Plot Waddington potential in 3D space.

        Parameters
        ----------
        waddington_potential : np.ndarray
            Waddington potential of each sample.
            This array should be calculated by 'calculate_waddington_potential' method.

        mode : {'pca', 'umap'}, optional
            The space to plot Waddington potential, by default "pca"

        gene_name : str, optional
            Gene name to color the points, by default None
            If None, the points will be colored by Waddington potential.
            If specified, the points will be colored by the expression of the specified gene.

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './waddington_potential.html'
        """
        
        if save and save_path is None:
            save_path = "./waddington_potential.html"

        X_concated = pd.concat(self.X_pca[:-1] if mode == "pca" else self.X_umap[:-1])
        if gene_name is None:
            color = waddington_potential
        else:
            color = pd.concat(self.X_selected)[: len(waddington_potential)][gene_name]
        plot_data = pd.DataFrame(index=X_concated.index)
        plot_data["x"] = X_concated.iloc[:, 0]
        plot_data["y"] = X_concated.iloc[:, 1]
        plot_data["z"] = waddington_potential
        fig = px.scatter_3d(
            plot_data,
            x="x",
            y="y",
            z="z",
            color=color,
            hover_name=X_concated.index,
            width=1000,
            height=800,
            labels={
                "x": X_concated.columns.values[0],
                "y": X_concated.columns.values[1],
                "z": "Waddington Potential",
            },
        )
        fig.update_traces(marker_size=1)
        if gene_name is None:
            fig.update_layout(title="Waddington potential")
        else:
            fig.update_layout(title=f"Waddington potential, gene = {gene_name}")
        fig.show()

        if save:
            fig.write_html(save_path)

    def plot_waddington_potential_surface(
        self,
        waddington_potential,
        mode="pca",
        save=False,
        save_path=None,
    ):
        """
        Plot Waddington's landscape in 3D space by using cellmap.

        Parameters
        ----------
        waddington_potential : np.ndarray
            Waddington potential of each sample.
            This array should be calculated by 'calculate_waddington_potential' method
            
        mode : {'pca', 'umap'}, optional
            The space to plot Waddington potential, by default "pca"    

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './wadding_potential_surface.html'
        """
        
        if save and save_path is None:
            save_path = "./wadding_potential_surface"
        if save_path is not None and save_path.split(".")[-1] == "html":
            save_path = save_path[:-5]

        day_labels = list(
            itertools.chain.from_iterable(
                [
                    [f"{str(self.day_names[i])}"] * len(self.X_pca[i])
                    for i in range(len(self.X_pca) - 1)
                ]
            )
        )
        adata_cellmap = anndata.AnnData(
            pd.concat(self.X_pca[:-1] if mode == "pca" else self.X_umap[:-1]),
            obs=pd.DataFrame(
                {
                    "clusters": day_labels,
                    "n_counts": np.sum(
                        pd.concat(
                            self.X_pca[:-1] if mode == "pca" else self.X_umap[:-1]
                        ),
                        axis=1,
                    ),
                    "potential": waddington_potential,
                },
                index=pd.concat(self.X_pca[:-1]).index,
            ),
            obsm={
                "X_show": pd.concat(
                    self.X_pca[:-1] if mode == "pca" else self.X_umap[:-1]
                )
                .iloc[:, :2]
                .values,
            },
        )

        cellmap.view_3D(
            adata_cellmap,
            basis="show",
            show_shadow=False,
            save=save,
            filename=save_path,
        )

    def bures_wasserstein_distance(self, m_0, m_1, sigma_0, sigma_1):
        sigma_00 = spl.sqrtm(sigma_0)
        sigma_010 = spl.sqrtm(sigma_00 @ sigma_1 @ sigma_00)
        d = np.linalg.norm(m_0 - m_1) ** 2 + np.trace(sigma_0 + sigma_1 - 2 * sigma_010)
        return d

    def egot(
        self,
        pi_0,
        pi_1,
        mu_0,
        mu_1,
        S_0,
        S_1,
        reg=0.01,
        numItermax=5000,
        method="sinkhorn_epsilon_scaling",
        tau=1e8,
        stopThr=1e-9,
        sinkhorn_other_params={},
    ):
        K_0 = mu_0.shape[0]
        K_1 = mu_1.shape[0]
        d = mu_0.shape[1]
        S_0 = S_0.reshape(K_0, d, d)
        S_1 = S_1.reshape(K_1, d, d)
        M = np.zeros((K_0, K_1))
        for k in range(K_0):
            for l in range(K_1):
                M[k, l] = self.bures_wasserstein_distance(
                    mu_0[k, :], mu_1[l, :], S_0[k, :, :], S_1[l, :, :]
                )
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings(
                "always",
                message="Sinkhorn did not converge.",
                category=UserWarning,
                module="ot"    
            )
            solution = ot.sinkhorn(
                pi_0,
                pi_1,
                M / M.max(),
                reg=reg,
                numItermax=numItermax,
                method=method,
                tau=tau,
                stopThr=stopThr,
                **sinkhorn_other_params,
            )

        if len(w):
            warnings.warn(
                "Warning: Sinkhorn did not converge.\n"
                "You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`.",
                UserWarning
            )

        return solution

    def calculate_solution(
        self,
        gmm_source,
        gmm_target,
        reg=0.01,
        numItermax=5000,
        method="sinkhorn_epsilon_scaling",
        tau=1e8,
        stopThr=1e-9,
        sinkhorn_other_params={},
    ):
        pi_0, pi_1 = gmm_source.weights_, gmm_target.weights_
        mu_0, mu_1 = gmm_source.means_, gmm_target.means_
        S_0, S_1 = gmm_source.covariances_, gmm_target.covariances_

        solution = self.egot(
            pi_0,
            pi_1,
            mu_0,
            mu_1,
            S_0,
            S_1,
            reg,
            numItermax,
            method,
            tau,
            stopThr,
            sinkhorn_other_params,
        )
        return solution

    def calculate_solutions(
        self,
        gmm_models,
        reg=0.01,
        numItermax=5000,
        method="sinkhorn_epsilon_scaling",
        tau=1e8,
        stopThr=1e-9,
        sinkhorn_other_params={},
    ):
        solutions = []
        for i in range(len(gmm_models) - 1):
            solutions.append(
                self.calculate_solution(
                    gmm_models[i],
                    gmm_models[i + 1],
                    reg,
                    numItermax,
                    method,
                    tau,
                    stopThr,
                    sinkhorn_other_params,
                )
            )
        return solutions

    def calculate_normalized_solutions(
        self,
        gmm_models,
        reg=0.01,
        numItermax=5000,
        method="sinkhorn_epsilon_scaling",
        tau=1e8,
        stopThr=1e-9,
        sinkhorn_other_params={},
    ):
        solutions_normalized = []
        for i in range(len(gmm_models) - 1):
            solution = self.calculate_solution(
                gmm_models[i],
                gmm_models[i + 1],
                reg,
                numItermax,
                method,
                tau,
                stopThr,
                sinkhorn_other_params,
            )
            solutions_normalized.append((solution.T / gmm_models[i].weights_).T)
        return solutions_normalized

    def gaussian_mixture_density(self, mu, sigma, alpha, x):
        K = mu.shape[0]
        alpha = alpha.reshape(1, K)
        y = 0
        for j in range(K):
            y += alpha[0, j] * multivariate_normal.pdf(
                x, mean=mu[j, :], cov=sigma[j, :, :]
            )
        return y

    def calculate_mut_st(self, gmm_source, gmm_target, t):
        d = gmm_source.means_.shape[1]
        K_0, K_1 = gmm_source.means_.shape[0], gmm_target.means_.shape[0]

        mu_0, mu_1 = gmm_source.means_, gmm_target.means_
        S_0, S_1 = gmm_source.covariances_, gmm_target.covariances_

        mut = np.zeros((K_0 * K_1, d))
        St = np.zeros((K_0 * K_1, d, d))
        for k in range(K_0):
            for l in range(K_1):
                mut[k * K_1 + l, :] = (1 - t) * mu_0[k, :] + t * mu_1[l, :]
                sigma_1 = spl.sqrtm(S_1[l, :, :])
                B = (
                    sigma_1
                    @ spl.inv(spl.sqrtm(sigma_1 @ S_0[k, :, :] @ sigma_1))
                    @ sigma_1
                )
                St[k * K_1 + l, :, :] = (
                    ((1 - t) * np.eye(d) + t * B)
                    @ S_0[k, :, :]
                    @ ((1 - t) * np.eye(d) + t * B)
                )

        return mut, St

    def generate_cluster_names_with_day(self, cluster_names=None):
        if cluster_names is None:
            cluster_names = []
            if self.gmm_label_converter is None:
                for i in range(len(self.gmm_n_components_list)):
                    cluster_names.append(
                        [f"{j}" for j in range(self.gmm_n_components_list[i])]
                    )
            else:
                cluster_names = self.gmm_label_converter

        cluster_names_with_day = []
        for i in range(len(self.day_names)):
            cluster_names_with_day.append(
                [f"{self.day_names[i]}-{cluster}" for cluster in cluster_names[i]]
            )
        return cluster_names_with_day

    def get_gmm_means(self):
        gmm_means = []
        for gmm_model in self.gmm_models:
            gmm_means.append(gmm_model.means_)
        return gmm_means

    def _get_gmm_mean_gene_values_per_cluster(self, gmm_means, cluster_names=None):
        if cluster_names is None:
            cluster_names = list(range(sum(self.gmm_n_components_list)))

        gmm_mean_gene_values_per_cluster = list(
            itertools.chain.from_iterable(
                self.pca_model.inverse_transform(gmm_means[i])
                for i in range(len(gmm_means))
            )
        )
        gmm_mean_gene_values_per_cluster = pd.DataFrame(
            gmm_mean_gene_values_per_cluster,
            index=cluster_names,
            columns=self.gene_names,
        )
        return gmm_mean_gene_values_per_cluster

    def get_positive_gmm_mean_gene_values_per_cluster(
        self, gmm_means, cluster_names=None,
    ):
        gmm_mean_gene_values_per_cluster = self._get_gmm_mean_gene_values_per_cluster(
            gmm_means, cluster_names
        )
        gmm_mean_gene_values_per_cluster = gmm_mean_gene_values_per_cluster.where(
            gmm_mean_gene_values_per_cluster > 0, 0
        )
        return gmm_mean_gene_values_per_cluster

    def replace_gmm_labels(self, converter):
        gmm_labels_modified = []
        for i in range(len(self.gmm_labels)):
            gmm_labels_modified.append(
                np.array([converter[i][label] for label in self.gmm_labels[i]])
            )
        self.gmm_labels_modified = gmm_labels_modified
        self.gmm_label_converter = converter

    def create_separated_data(
        self,
        data_names,
        min_cluster_size=2,
        return_cluster_names=False,
        cluster_names=None,
        original_covariances_weight = 0
    ):
        """
        Create separated data for each data name.

        Parameters
        ----------
        data_names : list of str
            List of prefixes to identify datasets.
            Cells with names starting with these strings will be extracted into separate scEGOT objects.

        min_cluster_size : int, optional
            Minimum number of cells required to retain a cluster, by default 2.
            Clusters smaller than this threshold will be removed in the separated objects.

        return_cluster_names : bool, optional
            If True, also return the cluster names for each separated object, by default False.

        cluster_names : list of list of str, optional
            Custom names for the clusters in the original object, by default None.
            1st dimension is the number of days, 2nd dimension is the number of gmm components
            in each day.
            If None, names are automatically generated by generate_cluster_names_with_day() method.
            
        original_covariances_weight : float, optional
            Weight factor for blending the original GMM covariances with recalculated ones, by default 0.
            The new covariance is calculated as:
            new_cov = original_cov * weight + recalculated_cov * (1 - weight).
            - 0.0: Use only covariances calculated from the separated data.
            - 1.0: Use only covariances of the original object.

        Returns
        -------
        dict
            A dictionary where keys are `data_names` and values are the corresponding separated `scEGOT` objects.

        dict
            A dictionary where keys are `data_names` and values are lists of cluster names.
            These names correspond to the cluster names of the original object.
            This will be returned only when 'return_cluster_names' is True.
        """

        separated_scegot_dict = {}
        separated_cluster_names_dict = {}
        umap_flag = self.X_umap is not None and self.umap_model is not None
        gmm_model_flag = self.gmm_models is not None
        gmm_label_flag = self.gmm_labels is not None
        return_cluster_names = gmm_label_flag and return_cluster_names

        if cluster_names is None:
            cluster_names = self.generate_cluster_names_with_day()

        for data_name in data_names:
            separated_X_raw = []
            separated_X_normalized = []
            separated_X_selected = []
            separated_X_pca = []
            separated_gmm_models = []
            separated_gmm_labels = []
            if umap_flag:
                separated_X_umap = []
            if gmm_label_flag:
                gmm_n_components_list = self.gmm_n_components_list.copy()
                removed_cluster_names = []
            if return_cluster_names:
                separated_cluster_names = copy.deepcopy(cluster_names) 

            for day in range(len(self.day_names)):
                day_separated_X_raw = self.X_raw[day].loc[self.X_raw[day].index.str.startswith(data_name)]
                day_separated_X_normalized = self.X_normalized[day].loc[self.X_normalized[day].index.str.startswith(data_name)]
                day_separated_X_selected = self.X_selected[day].loc[self.X_selected[day].index.str.startswith(data_name)]
                day_separated_X_pca = self.X_pca[day].loc[self.X_pca[day].index.str.startswith(data_name)]

                if umap_flag:
                    day_separated_X_umap = self.X_umap[day].loc[self.X_umap[day].index.str.startswith(data_name)]
                    separated_X_umap.append(day_separated_X_umap)
                
                if gmm_model_flag:
                    day_concated_gmm_model = self.gmm_models[day]

                    if gmm_label_flag:
                        day_separated_gmm_label = self.gmm_labels[day][self.X_raw[day].index.str.startswith(data_name)]
                        day_separated_gmm_model = sklearn_clone(day_concated_gmm_model)
                        cluster_sizes = []
                        means = []
                        covariances = []
                        precisions = []
                        precisions_cholesky = []
                        removed_clusters_num = 0
                        for cluster_index in range(day_separated_gmm_model.n_components):
                            cluster_data = day_separated_X_pca[day_separated_gmm_label == (cluster_index - removed_clusters_num)]
                            n_cluster_data_rows = cluster_data.shape[0]
                            if n_cluster_data_rows < min_cluster_size:
                                if self.verbose:
                                    removed_cluster_names.append(cluster_names[day][cluster_index])
                                gmm_n_components_list[day] -= 1
                                day_separated_gmm_model.n_components -= 1
                                if n_cluster_data_rows >= 1:
                                    not_removed_cell_mask = day_separated_gmm_label != (cluster_index - removed_clusters_num)
                                    day_separated_X_raw = day_separated_X_raw.loc[not_removed_cell_mask]
                                    day_separated_X_normalized = day_separated_X_normalized.loc[not_removed_cell_mask]
                                    day_separated_X_selected = day_separated_X_selected.loc[not_removed_cell_mask]
                                    day_separated_X_pca = day_separated_X_pca.loc[not_removed_cell_mask]
                                    day_separated_gmm_label = day_separated_gmm_label[not_removed_cell_mask]
                                    if umap_flag:
                                        day_separated_X_umap = day_separated_X_umap.loc[not_removed_cell_mask]
                                day_separated_gmm_label = np.where(
                                    day_separated_gmm_label > cluster_index, day_separated_gmm_label - 1, day_separated_gmm_label
                                )
                                if return_cluster_names:
                                    del separated_cluster_names[day][cluster_index - removed_clusters_num]
                                removed_clusters_num += 1
                                continue

                            if n_cluster_data_rows <= self.pca_model.n_components_ and original_covariances_weight == 0:
                                    msg = (
                                        f"The number of cells in the cluster {cluster_names[day][cluster_index]} "
                                        f"for {data_name} ({n_cluster_data_rows}) is less than or equal to "
                                        f"the number of PCA components ({self.pca_model.n_components_}). "
                                        "The covariance matrix cannot be inverted accurately."
                                    )
                                    raise ValueError(msg)
                                
                            cluster_sizes.append(len(cluster_data))
                            means.append(cluster_data.mean().values)
                            
                            original_cov = day_concated_gmm_model.covariances_[cluster_index]
                            cov = original_cov * original_covariances_weight + np.cov(cluster_data.T) * (1 - original_covariances_weight)
                            cov_cholesky = np.linalg.cholesky(cov)
                            prec_cholesky = np.linalg.solve(cov_cholesky, np.eye(cov.shape[0], dtype=cov.dtype)).T
                            prec = np.dot(prec_cholesky, prec_cholesky.T)

                            covariances.append(cov)
                            precisions.append(prec)
                            precisions_cholesky.append(prec_cholesky)
                                
                        separated_gmm_labels.append(day_separated_gmm_label)

                        day_separated_gmm_model.weights_ = np.array(cluster_sizes) / len(day_separated_X_raw)
                        day_separated_gmm_model.means_ = np.array(means)
                        day_separated_gmm_model.covariances_ = np.array(covariances)
                        day_separated_gmm_model.precisions_ = np.array(precisions)
                        day_separated_gmm_model.precisions_cholesky_ = np.array(precisions_cholesky)
                        day_separated_gmm_model.converged_ = day_concated_gmm_model.converged_
                        day_separated_gmm_model.lower_bound_ = day_concated_gmm_model.lower_bound_
                        day_separated_gmm_model.n_features_in_ = day_concated_gmm_model.n_features_in_
                        day_separated_gmm_model.n_iter_ = day_concated_gmm_model.n_iter_
                        if hasattr(day_concated_gmm_model, "lower_bounds_"):
                            day_separated_gmm_model.lower_bounds_ = day_concated_gmm_model.lower_bounds_

                        separated_gmm_models.append(day_separated_gmm_model)
                    else:
                        separated_gmm_models.append(sklearn_clone(day_concated_gmm_model))
                
                separated_X_raw.append(day_separated_X_raw)
                separated_X_normalized.append(day_separated_X_normalized)
                separated_X_selected.append(day_separated_X_selected)
                separated_X_pca.append(day_separated_X_pca)
            
            if len(removed_cluster_names) > 0:
                msg = (
                    f"The following clusters in {data_name} have been removed because "
                    f"the number of cells is less than the minimum cluster size ({min_cluster_size}): \n"
                    f"{', '.join(removed_cluster_names)}."
                )
                print("Info: \n" + msg)

            separated_scegot = scEGOT(separated_X_raw, day_names=self.day_names, verbose=self.verbose)
            separated_scegot.X_normalized = separated_X_normalized
            separated_scegot.X_selected = separated_X_selected
            separated_scegot.X_pca = separated_X_pca
            
            separated_scegot.pca_model = copy.deepcopy(self.pca_model)
            separated_scegot.gene_names = self.gene_names.copy()
            separated_scegot.gmm_label_converter = copy.deepcopy(self.gmm_label_converter)

            if umap_flag:
                separated_scegot.X_umap = separated_X_umap
                separated_scegot.umap_model = copy.deepcopy(self.umap_model)

            if gmm_model_flag:
                separated_scegot.gmm_n_components_list = gmm_n_components_list
                separated_scegot.gmm_models = separated_gmm_models
            
            if gmm_label_flag:
                separated_scegot.gmm_labels = separated_gmm_labels
                separated_scegot.gmm_labels_modified = separated_gmm_labels

            separated_scegot_dict[data_name] = separated_scegot

            if return_cluster_names:
                separated_cluster_names_dict[data_name] = separated_cluster_names

        if return_cluster_names:
            return separated_scegot_dict, separated_cluster_names_dict
        else:
            return separated_scegot_dict


class CellStateGraph():
    def __init__(
        self,
        G, 
        scegot,
        threshold=0.05,
        mode="pca",
        cluster_names=None,
        node_ids=None,
        merge_clusters_by_name=False,
        x_reverse=False,
        y_reverse=False,
        require_parent=False
    ):
        self.G = G
        self.scegot = scegot
        self.threshold = threshold
        self.mode = mode
        self.cluster_names = cluster_names
        self.node_ids = node_ids
        self.merge_clusters_by_name = merge_clusters_by_name
        self.x_reverse = x_reverse
        self.y_reverse = y_reverse
        self.require_parent = require_parent
        self.day_num = len(scegot.day_names)
        self.gmm_n_components_list = scegot.gmm_n_components_list
        
    def reverse_graph(self, x=False, y=False):
        """
        Reverse the graph layout along the specified axes.

        Parameters
        ----------
        x : bool, optional
            If True, reverse the x-axis of the graph layout, by default False.
        
        y : bool, optional
            If True, reverse the y-axis of the graph layout, by default False.
        """

        if x:
            self.x_reverse = not self.x_reverse
            for node in self.G.nodes():
                self.G.nodes[node]["pos"] = (-1 * self.G.nodes[node]["pos"][0], self.G.nodes[node]["pos"][1])
        if y:
            self.y_reverse = not self.y_reverse
            for node in self.G.nodes():
                self.G.nodes[node]["pos"] = (self.G.nodes[node]["pos"][0], -1 * self.G.nodes[node]["pos"][1])
    
    def _validate_cluster_names(self, cluster_names):
        if type(cluster_names) != list:
            raise TypeError("The type of 'cluster_names' should be list.")
        day_num = self.day_num
        if len(cluster_names) != day_num:
            raise ValueError(f"The length of 'cluster_names' should be equal to the number of days ({day_num}).")
        for day in range(day_num):
            if type(cluster_names[day]) != list:
                raise TypeError(f"The element located at index {day} in 'cluster_names' should be a list.")
            if len(cluster_names[day]) != self.gmm_n_components_list[day]:
                raise ValueError(
                    f"The element located at index {day} in 'cluster_names' must contain the same number of elements as "
                    f"the number of clusters of the {self.scegot.day_names[day]} (= {self.gmm_n_components_list[day]}), \n"
                    f"The actual length of the element at index {day} in 'cluster_names' was {len(cluster_names[day])}."
                )
        if self.merge_clusters_by_name:
            cluster_names_flattened = list(itertools.chain.from_iterable(cluster_names))
            old_cluster_names_flattened = list(itertools.chain.from_iterable(self.cluster_names))
            id_name_dict = {}
            for i in range(len(self.node_ids)):
                id = self.node_ids[i]
                name = cluster_names_flattened[i]
                if id in id_name_dict:
                    if id_name_dict[id] != name:
                        raise ValueError(
                            f"When merge_clusters_by_name = True, clusters that shared "
                            f"the same original name must be given the same new name.\n"
                            f"Cluster '{old_cluster_names_flattened[i]}' has inconsistent "
                            f"names: '{id_name_dict[id]}' and '{name}'."
                        )
                else:
                    id_name_dict[id] = name
        return cluster_names
    
    def set_cluster_names(self, cluster_names):
        """
        Set new cluster names for the cell state graph.  

        Parameters
        ----------
        cluster_names : list of list of str
            New names for the clusters.
            1st dimension is the number of days, 2nd dimension is the number of gmm components
            in each day.
            Merged clusters must have the same name when 'merge_clusters_by_name' is True.
        Returns
        -------
        list of list of str
            The new cluster names.
        """

        new_cluster_names = self._validate_cluster_names(cluster_names)
        self.cluster_names = new_cluster_names
        return new_cluster_names
    
    def _day_update_cluster_names(self, cluster_names, cluster_names_map, day):
        for cluster_num in range(self.gmm_n_components_list[day]):
            old_name = cluster_names[day][cluster_num]
            if old_name in cluster_names_map.keys():
                cluster_names[day][cluster_num] = cluster_names_map[old_name]
        return cluster_names

    def update_cluster_names(self, cluster_names_map, day=None):
        """
        Update cluster names for the cell state graph based on a mapping dictionary.

        Parameters
        ----------
        cluster_names_map : dict
            A dictionary mapping old cluster names to new cluster names.
        day : int, optional
            The specific day to update cluster names for, by default None.
            If None, update cluster names for all days.
        
        Returns
        -------
        list of list of str
            The updated cluster names.
        """

        cluster_names = copy.deepcopy(self.cluster_names)
        if day is None:
            for day in range(self.day_num):
                cluster_names = self._day_update_cluster_names(cluster_names, cluster_names_map, day)
        else:
            cluster_names = self._day_update_cluster_names(cluster_names, cluster_names_map, day)
        new_cluster_names = self._validate_cluster_names(cluster_names)
        self.cluster_names = new_cluster_names
        return new_cluster_names

    def _get_day_node_dict(self):
        day_dict = dict(self.G.nodes(data="day"))
        day_node_dict = defaultdict(list)

        for cluster, day in day_dict.items():
            day_node_dict[day].append(cluster)

        return day_node_dict

    def _get_node_name_alphabetical_order_dict(self):
        cluster_alphabetical_order = {}
        for day_cluster_names in self.cluster_names:
            sorted_day_cluster_names = sorted(set(day_cluster_names))
            for order, name in enumerate(sorted_day_cluster_names):
                cluster_alphabetical_order[name] = order
        return cluster_alphabetical_order
    
    def _get_node_position_dict(self, layout, y_position):
        G = self.G
        pos = {}

        if layout == "normal":
            pos = {node: G.nodes[node]["pos"] for node in G.nodes()}
        else:
            if y_position == "weight":
                for node in G.nodes():
                    pos[node] = (G.nodes[node]["day"], -G.nodes[node]["cluster_weight"])
            else:
                if y_position == "name":
                    ypos_dict = self._get_node_name_alphabetical_order_dict()
                else:
                    ypos_dict = y_position
                for node in G.nodes():
                    node_day = G.nodes[node]["day"]
                    node_gmm = G.nodes[node]["cluster_gmm_list"][0]
                    node_name = self.cluster_names[node_day][node_gmm]
                    try:
                        ypos = -ypos_dict[node_name]
                    except:
                        raise ValueError(f"The node name '{node_name}' does not exist in 'y_position'.")
                    pos[node] = (G.nodes[node]["day"], ypos)

        return pos

    def plot_simple_cell_state_graph(
        self,
        layout="normal",
        y_position="name",
        cluster_names=None,
        node_weight_annotation=False,
        edge_weight_annotation=False,
        save=False,
        save_path=None
    ):
        """
        Plot the cell state graph with the given graph object in a simple way.

        Parameters
        ----------
        layout : {'normal', 'hierarchy'}, optional
            The layout of the graph, by default "normal".

            * When "normal", the graph is plotted in PCA or UMAP space.
            * When "hierarchy", the graph is plotted with the day on the x-axis and the cluster on the y-axis.

        y_position : str or dict, optional
            Determines the y-axis position of nodes when layout is "hierarchy", by default "name".

            * "name": Sort nodes alphabetically by name.
            * "weight": Sort nodes by their weight.
            * dict: A dictionary mapping node names to y-axis positions.

            This parameter is ignored when layout is "normal".

        cluster_names : list of list of str, optional
            Custom names for the clusters, by default None.

            1st dimension is the number of days, 2nd dimension is the number of gmm components
            in each day.
            When the attribute ``merge_clusters_by_name`` is True, clusters to be merged must be given
            the same new name.

            If None, the attribute ``cluster_names`` is used.

        node_weight_annotation : bool, optional
            If True, display the weight of each node, by default False.

        edge_weight_annotation : bool, optional
            If True, display the weight of each edge, by default False.

        save : bool, optional
            If True, save the output image, by default False.

        save_path : str, optional
            Path to save the output image, by default None.
            If None, the image will be saved as './simple_cell_state_graph.png'.

        Raises
        ------
        ValueError
            This error is raised in the following cases:
            - When 'layout' is not 'normal' or 'hierarchy'.
            - When 'y_position' is a string but not 'name' or 'weight'.
        TypeError
            When 'y_position' is not a string or dict (if layout is 'hierarchy').
        """
        
        if layout not in ["normal", "hierarchy"]:
            raise ValueError("The parameter 'layout' should be 'normal or 'hierarchy'.")
        if layout == "hierarchy":
            if type(y_position) not in [str, dict]:
                raise TypeError("The Type of 'y_position' should be string or dict.")
            if type(y_position) == str and y_position not in ["name", "weight"]:
                raise ValueError(
                    "The parameter 'y_position' should be 'name', 'weight' or dictionary object."
                )
        
        if cluster_names is None:
            cluster_names = self.cluster_names
        else:
            cluster_names = self._validate_cluster_names(cluster_names)

        if save and save_path is None:
            save_path = "./simple_cell_state_graph.png"

        G = self.G

        node_color = [node["day"] for node in G.nodes.values()]
        edge_color = np.array([G.edges[edge]["weight"] for edge in G.edges()])
        pos = self._get_node_position_dict(layout, y_position)
        fig, ax = plt.subplots(figsize=(12, 10))
        
        nx.draw(
            G,
            pos,
            node_size=[node["weight"] * 4500 for node in G.nodes.values()],
            node_color="white",
            edge_color="black",
            arrows=True,
            arrowsize=30,
            linewidths=2,
            ax=ax,
            width=6.0,
        )
        nx.draw(
            G,
            pos,
            node_size=[node["weight"] * 5000 for node in G.nodes.values()],
            node_color="white",
            edge_color="white",
            arrows=True,
            arrowsize=30,
            linewidths=2,
            ax=ax,
            width=5.0,
        )

        node_cmap = (
            plt.cm.tab10(np.arange(10))
            if self.day_num <= 10
            else plt.cm.tab20(np.arange(20))
        )
        nx.draw(
            G,
            pos,
            node_size=[node["weight"] * 5000 for node in G.nodes.values()],
            node_color=node_color,
            edge_color=edge_color,
            edgecolors="white",
            arrows=True,
            arrowsize=30,
            linewidths=2,
            cmap=ListedColormap(node_cmap[: self.day_num]),
            edge_cmap=plt.cm.Reds,
            ax=ax,
            alpha=1,
            width=5.0,
        )

        if edge_weight_annotation: 
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels={edge: f"{G.edges[edge]['weight']:.3f}" for edge in G.edges()},
                font_size=14,
                label_pos=0.3,
                ax=ax
            )

        texts = []
        for node in G.nodes():
            node_day = G.nodes[node]["day"]
            node_gmm = G.nodes[node]["cluster_gmm_list"][0]
            node_name = cluster_names[node_day][node_gmm]
            text_ = ax.text(
                pos[node][0],
                pos[node][1],
                f'{node_name}\n{G.nodes[node]["weight"]:.3f}' if node_weight_annotation else node_name,
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
            )
            text_.set_path_effects(
                [patheffects.withStroke(linewidth=3, foreground="w")]
            )
            texts.append(text_)

        if layout == "normal":
            adjust_text(texts)

        plt.show()

        if save:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")


    def _calculate_weighted_mean_of_gene_values(self, df):
        if len(df) == 1:
            return df.drop("weights", axis=1)
        else:
            weighted_df = df.drop("weights", axis=1).mul(df["weights"], axis=0)
            return (weighted_df.sum() / df["weights"].sum()).to_frame(name=df.index[0]).T
        
    def _get_up_regulated_genes(self, gene_values, num=10):
        scegot = self.scegot
        columns = [f"up_gene_{i+1}" for i in range(num)]
        df_upgenes = pd.DataFrame(columns=columns, index=pd.MultiIndex.from_tuples([], names=["source", "target"]))
        for edge in self.G.edges():
            fold_change = scegot._get_fold_change(
                gene_values,
                edge[0],
                edge[1],
            )
            upgenes = pd.DataFrame(
                [scegot._get_nlargest_gene_indices(fold_change, num=num).values],
                columns=columns,
                index=pd.MultiIndex.from_tuples([(str(edge[0]), str(edge[1]))], names=["source", "target"])
            )
            df_upgenes = pd.concat([df_upgenes, upgenes])
        return df_upgenes

    def _get_down_regulated_genes(self, gene_values, num=10):
        scegot = self.scegot
        columns = [f"down_gene_{i+1}" for i in range(num)]
        df_downgenes = pd.DataFrame(columns=columns, index=pd.MultiIndex.from_tuples([], names=["source", "target"]))
        for edge in self.G.edges():
            fold_change = scegot._get_fold_change(
                gene_values,
                edge[0],
                edge[1],
            )
            downgenes = pd.DataFrame(
                [scegot._get_nsmallest_gene_indices(fold_change, num=num).values],
                columns=columns,
                index=pd.MultiIndex.from_tuples([(str(edge[0]), str(edge[1]))], names=["source", "target"])
            )
            df_downgenes = pd.concat([df_downgenes, downgenes])
        return df_downgenes

    def _get_genes_info(self, gene_names, gene_pick_num):
        scegot = self.scegot
        
        mean_gene_values_per_cluster = (
            scegot.get_positive_gmm_mean_gene_values_per_cluster(
                scegot.get_gmm_means(),
                self.node_ids,
            )
        )
        if self.merge_clusters_by_name:
            cluster_weights = pd.Series(scegot._get_gmm_node_weights_flattened(), index=mean_gene_values_per_cluster.index, name="weight")
            mean_gene_values_per_cluster["weights"] = cluster_weights
            mean_gene_values_per_cluster = mean_gene_values_per_cluster.groupby(level=0).apply(self._calculate_weighted_mean_of_gene_values)
            mean_gene_values_per_cluster = mean_gene_values_per_cluster.reset_index(level=1, drop=True)
        
        mean_gene_values_per_cluster = mean_gene_values_per_cluster.loc[
            :, mean_gene_values_per_cluster.columns.isin(gene_names)
        ]

        nlargest_genes = mean_gene_values_per_cluster.T.apply(
            scegot._get_nlargest_gene_indices, num=gene_pick_num
        ).T
        nsmallest_genes = mean_gene_values_per_cluster.T.apply(
            scegot._get_nsmallest_gene_indices, num=gene_pick_num
        ).T
        nlargest_genes.columns += 1
        nsmallest_genes.columns += 1

        up_genes = self._get_up_regulated_genes(
            mean_gene_values_per_cluster, num=gene_pick_num
        )
        down_genes = self._get_down_regulated_genes(
            mean_gene_values_per_cluster, num=gene_pick_num
        )

        return nlargest_genes, nsmallest_genes, up_genes, down_genes

    def plot_cell_state_graph(
        self,
        layout="normal",
        y_position="name",
        cluster_names=None,
        gene_names=None,
        gene_pick_num=5,
        plot_title="Cell State Graph",
        save=False,
        save_path=None,
    ):
        """
        Plot the cell state graph with the given graph object.

        Parameters
        ----------
        layout : {'normal', 'hierarchy'}, optional
            The layout of the graph, by default "normal"

            * When 'normal', the graph is plotted in PCA or UMAP space.
            * When 'hierarchy', the graph is plotted with the day on the x-axis and the cluster on the y-axis.

        y_position : str or dict, optional
            Determines the y-axis position of nodes when layout is 'hierarchy', by default "name".

            * 'name': Sort nodes alphabetically by name.
            * 'weight': Sort nodes by their weight.
            * dict: A dictionary mapping node names to y-axis positions.

            This parameter is ignored when layout is 'normal'.

        cluster_names : list of list of str
            Custom names for the clusters, by default None.

            1st dimension is the number of days, 2nd dimension is the number of gmm components
            in each day.
            When the attribute ``merge_clusters_by_name`` is True, clusters to be merged must be given
            the same new name.

            If None, the attribute ``cluster_names`` is used.

        gene_names : list of str, optional
            List of gene names to use, by default None
            If None, all gene names (``self.scegot.gene_names``) will be used.
            You can pass on any list of gene names you want to use, not limited to TF genes.

        gene_pick_num : int, optional
            The number of genes to show in each node and edge, by default 5
        
        plot_title : str, optional
            Title of the plot, by default "Cell State Graph"

        save : bool, optional
            If True, save the output image, by default False

        save_path : str, optional
            Path to save the output image, by default None
            If None, the image will be saved as './cell_state_graph.png'
        """

        if layout not in ["normal", "hierarchy"]:
            raise ValueError("The parameter 'layout' should be 'normal or 'hierarchy'.")
        if layout == "hierarchy":
            if type(y_position) not in [str, dict]:
                raise TypeError("The Type of 'y_position' should be string or dict.")
            if type(y_position) == str and y_position not in ["name", "weight"]:
                raise ValueError("The parameter 'y_position' should be 'name', 'weight' or dictionary object.")

        if cluster_names is None:
            cluster_names = self.cluster_names
        else:
            cluster_names = self._validate_cluster_names(cluster_names)
        
        if gene_names is None:
            gene_names = self.scegot.gene_names

        if save and save_path is None:
            save_path = "./cell_state_graph.png"        

        nlargest_genes, nsmallest_genes, up_genes, down_genes = self._get_genes_info(gene_names, gene_pick_num)

        tail_list = []
        head_list = []
        color_list = []
        trace_recode = []

        G = self.G
        colors = plt.cm.inferno(np.linspace(0, 1, self.day_num + 2))
        pos = self._get_node_position_dict(layout, y_position)

        for edge in G.edges():
            x_0, y_0 = pos[edge[0]]
            x_1, y_1 = pos[edge[1]]
            tail_list.append((x_0, y_0))
            head_list.append((x_1, y_1))
            weight = G.edges[edge]["weight"] * 25
            color = colors[G.edges[edge]["color"] + 1]

            color_list.append(f"rgb({color[0]},{color[1]},{color[2]})")

            edge_trace = go.Scatter(
                x=tuple([x_0, x_1, None]),
                y=tuple([y_0, y_1, None]),
                mode="lines",
                line={"width": weight},
                line_color=f"rgb({color[0]},{color[1]},{color[2]})",
                line_shape="spline",
                opacity=0.4,
            )

            trace_recode.append(edge_trace)

        middle_hover_trace = go.Scatter(
            x=[],
            y=[],
            hovertext=[],
            mode="markers",
            textposition="top center",
            hoverinfo="text",
            marker={
                "size": 20,
                "color": [edge["color"] + 1 for edge in G.edges.values()],
            },
            opacity=0,
        )

        for edge in G.edges():
            source = G.nodes[edge[0]]
            target = G.nodes[edge[1]]
            x_0, y_0 = pos[edge[0]]
            x_1, y_1 = pos[edge[1]]
            genes_index = (str(edge[0]), str(edge[1]))
            source_day = source["day"]
            source_gmm = source["cluster_gmm_list"][0]
            source_name = cluster_names[source_day][source_gmm]
            target_day = target["day"]
            target_gmm = target["cluster_gmm_list"][0]
            target_name = cluster_names[target_day][target_gmm]
            hovertext = (
                f"<b>Edge from {source_name} to {target_name}</b><br>"
                f"weight = {G.edges[edge]['weight']:.4f}<br>"
                f"up_genes: {', '.join(up_genes.T[genes_index].values)}<br>"
                f"down_genes: {', '.join(down_genes.T[genes_index].values)}"
            )
            middle_hover_trace["x"] += tuple([(x_0 + x_1) / 2])
            middle_hover_trace["y"] += tuple([(y_0 + y_1) / 2])
            middle_hover_trace["hovertext"] += tuple([hovertext])
        
        trace_recode.append(middle_hover_trace)

        arrows = [
            go.layout.Annotation(
                dict(
                    x=head[0],
                    y=head[1],
                    showarrow=True,
                    xref="x",
                    yref="y",
                    arrowcolor=color,
                    arrowsize=2,
                    arrowwidth=2,
                    ax=tail[0],
                    ay=tail[1],
                    axref="x",
                    ayref="y",
                    arrowhead=1,
                )
            )
            for head, tail, color in zip(head_list, tail_list, color_list)
        ]
        
        node_x = []
        node_y = []
        node_names = []
        node_names_with_gmm_numbers = []
        node_gene_texts = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_day = G.nodes[node]["day"]
            node_gmm = G.nodes[node]["cluster_gmm_list"][0]
            node_name = cluster_names[node_day][node_gmm]
            node_cluster_gmm_list = G.nodes[node]["cluster_gmm_list"]
            node_names.append(node_name)
            node_names_with_gmm_numbers.append(
                f"<b>{node_name}</b><br>"
                f"weight = {G.nodes[node]['weight']:.4f}<br>"
                f"GMM cluster numbers = {', '.join(map(str, node_cluster_gmm_list))}"
            )
            node_gene_text = (
                f"<b>{node_name}</b><br>"
                f"largest_genes: {', '.join(nlargest_genes.T[node].values)}<br>"
                f"smallest_genes: {', '.join(nsmallest_genes.T[node].values)}"
            )
            node_gene_texts.append(node_gene_text)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=node_names,
            hovertext=node_names_with_gmm_numbers,
            textposition="top center",
            mode="markers+text",
            hoverinfo="text",
            marker=dict(line_width=2),
        )
        node_trace.marker.color = [node["day"] for node in G.nodes.values()]
        node_trace.marker.size = [node["weight"] * 140 for node in G.nodes.values()]
        trace_recode.append(node_trace)

        node_gene_trace = go.Scatter(
            x=node_x,
            y=node_y,
            hovertext=node_gene_texts,
            mode="markers",
            textposition="top center",
            hoverinfo="text",
            marker={
                "size": 20,
                "color": [node["day"] + 1 for node in G.nodes.values()],
            },
            opacity=0,
        )
        trace_recode.append(node_gene_trace)

        fig = go.Figure(
            data=trace_recode, layout=go.Layout(showlegend=False, hovermode="closest")
        )

        fig.update_layout(annotations=arrows)

        fig.update_layout(width=1000, height=800, title=plot_title)
        fig.show()

        if save:
            fig.write_image(save_path)