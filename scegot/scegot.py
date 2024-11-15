import itertools
import warnings
from io import BytesIO

import anndata
import cellmap
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydotplus
import scipy.linalg as spl
import screcode
import seaborn as sns
import umap.umap_ as umap
from adjustText import adjust_text
from IPython.display import HTML, Image, display
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from PIL import Image as PILImage
import scvelo as scv
from scanpy.pp import neighbors
from scipy import interpolate
from scipy.sparse import csc_matrix, issparse, lil_matrix, linalg
from scipy.stats import multivariate_normal, zscore
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import check_random_state
from tqdm import tqdm

sns.set_style("whitegrid")


def is_notebook() -> bool:
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


class scEGOT:
    def __init__(
        self,
        X,
        day_names=None,
        verbose=True,
        adata_day_key=None,
    ):
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

    def _select_highly_variable_genes(self, X_concated, n_select_genes=2000):
        genes = pd.DataFrame(index=X_concated.columns)
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
        highvar_genes = X_concated.loc[:, highvar_gene_names]
        return highvar_genes

    def _split_dataframe_by_row(self, df, row_counts):
        split_indices = list(itertools.accumulate(row_counts))
        df_list = [
            df.iloc[split_indices[i - 1] if i > 0 else 0 : split_indices[i]]
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
    ):
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
            X_concated = self._select_highly_variable_genes(X_concated, n_select_genes)

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
        return img.collections

    def animatie_interpolated_distribution(
        self,
        x_range=None,
        y_range=None,
        interpolate_interval=11,
        cmap="gnuplot2",
        save=False,
        save_path=None,
    ):
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

    def _get_cell_state_edge_list(self, cluster_names, thresh):
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
        if mode not in ["pca", "umap"]:
            raise ValueError("The parameter 'mode' should be 'pca' or 'umap'.")

        gmm_means_flattened = np.array(
            list(itertools.chain.from_iterable(self.get_gmm_means()))
        )
        if mode == "umap":
            gmm_means_flattened = self.umap_model.transform(gmm_means_flattened)

        cell_state_edge_list = self._get_cell_state_edge_list(cluster_names, threshold)
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
            hovertext = f"""up_genes: {', '.join(edges_up_gene.T[from_to].values)}<br>down_genes: {', '.join(edges_down_gene.T[from_to].values)}"""
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
            hovertext = f"""largest_genes: {', '.join(nodes_up_gene.T[node].values)}<br>smallest_genes: {', '.join(nodes_down_gene.T[node].values)}"""
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
        # nodes
        tf_nlargest = mean_tf_gene_values_per_cluster.T.apply(
            self._get_nlargest_gene_indices, num=tf_gene_pick_num
        ).T
        tf_nsmallest = mean_tf_gene_values_per_cluster.T.apply(
            self._get_nsmallest_gene_indices, num=tf_gene_pick_num
        ).T
        tf_nlargest.columns += 1
        tf_nsmallest.columns += 1
        # edges
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
        layout = "normal" or "hierarchy"
        order = None or "weight"
        """
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

        # draw edge border
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

        # draw edges
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
        tf_gene_names=None,
        save=False,
        save_path=None,
    ):
        if save and save_path is None:
            save_path = "./pathway_gene_expressions.png"

        if tf_gene_names is None:
            gene_names_to_use = self.gene_names
        else:
            gene_names_to_use = tf_gene_names

        genes = self.get_positive_gmm_mean_gene_values_per_cluster(
            self.get_gmm_means(),
            cluster_names=list(itertools.chain.from_iterable(cluster_names)),
        )
        genes = genes.loc[:, genes.columns.isin(gene_names_to_use)]

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
            "scegot.plot_pathway_single_gene_2d() will be depricated. Use scegot.plot_gene_expression_2d() instead.",
            FutureWarning,
        )
        self.plot_gene_expression_2d(gene_name, mode, col, save, save_path)

    def plot_gene_expression_2d(
        self, gene_name, mode="pca", col=None, save=False, save_path=None
    ):
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
            "scegot.plot_pathway_single_gene_3d() will be depricated. Use scegot.plot_gene_expression_3d() instead.",
            FutureWarning,
        )
        self.plot_gene_expression_3d(gene_name, col, save, save_path)

    def plot_gene_expression_3d(self, gene_name, col=None, save=False, save_path=None):
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
        color_points = "gmm", "day", or None
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
        color_points = "gmm", "day", or None
        """
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
        waddington_potential, *_ = linalg.lsqr(lap, F_all)

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
        numItermax=int(1e10),
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="ot")
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
        return solution

    def calculate_solution(
        self,
        gmm_source,
        gmm_target,
        reg=0.01,
        numItermax=int(1e10),
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
        numItermax=int(1e10),
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
        numItermax=int(1e10),
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
        self, gmm_means, cluster_names=None
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
