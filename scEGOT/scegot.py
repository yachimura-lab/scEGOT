# import warnings
import itertools
import ot
import numpy as np
import pandas as pd
from scipy import interpolate
import scipy.linalg as spl
from scipy.stats import multivariate_normal, zscore
from scipy.sparse import csr_matrix, csc_matrix, linalg
from sklearn import linear_model
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import Image, HTML, display
import screcode
import seaborn as sns
import pydotplus
from tqdm import tqdm

sns.set_style("whitegrid")
# warnings.filterwarnings("ignore")

class scEGOT:
    def __init__(
        self,
        X,
        pca_n_components,
        gmm_n_components_list,
        day_names,
        *,
        umap_n_components=None,
        umap_n_neighbors=None,
        verbose=True,
        umi_norm=1e4,
    ):
        self.pca_n_components = pca_n_components
        self.gmm_n_components_list = gmm_n_components_list

        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.verbose = verbose

        self.umi_norm = umi_norm

        self.X_raw = [df.copy() for df in X]
        self.X_PCA = None
        self.X_UMAP = None
        self.X_normalized = None

        self.pca_model = None
        self.gmm_models = None
        self.gmm_labels = None
        self.umap_model = None

        self.gene_names = X[0].columns
        self.day_names = day_names

        self.solutions = None

    def _preprocess_recode(self, X_concated, random_state=None):
        X_concated = pd.DataFrame(
            screcode.RECODE(
                verbose=self.verbose,
                stat_learning=True,
                stat_learning_seed=random_state,
            ).fit_transform(X_concated.values),
            index=X_concated.index,
            columns=X_concated.columns,
        )
        return X_concated

    def _preprocess_pca(self, X_concated, random_state=None):
        pca_model = PCA(n_components=self.pca_n_components, random_state=random_state)
        X_concated = pd.DataFrame(
            pca_model.fit_transform(X_concated.values),
            index=X_concated.index,
            columns=["PCA{}".format(i + 1) for i in range(self.pca_n_components)],
        )
        return X_concated, pca_model

    def _normalize_umi(self, X_concated):
        X_concated = X_concated.div(X_concated.sum(axis=1), axis=0) * self.umi_norm
        return X_concated

    def _normalize_log1p(self, X_concated):
        X_concated = X_concated.where(X_concated > 0, 0)
        X_concated = pd.DataFrame(
            np.log1p(X_concated.values),
            index=X_concated.index,
            columns=X_concated.columns,
        )
        return X_concated

    def _normalize_data(self, X_concated):
        X_concated = self._normalize_umi(X_concated)
        X_concated = self._normalize_log1p(X_concated)
        return X_concated

    def _split_dataframe_by_row(self, df, row_counts):
        split_indices = list(itertools.accumulate(row_counts))
        df_list = [
            df.iloc[split_indices[i - 1] if i > 0 else 0 : split_indices[i]]
            for i in range(len(split_indices))
        ]
        return df_list

    def preprocess(
        self,
        recode_random_state=None,
        pca_random_state=None,
        apply_recode=True,
        apply_normalization_log1p=True,
        apply_normalization_umi=True,
    ):
        if self.X_PCA is not None:
            return self.X_PCA, self.pca_model

        X_concated = pd.concat(self.X_raw)

        if apply_recode:
            if self.verbose:
                print("Applying scRECODE...")
            X_concated = self._preprocess_recode(X_concated, recode_random_state)

        if apply_normalization_umi:
            if self.verbose:
                print("Applying UMI normalization...")
            X_concated = self._normalize_umi(X_concated)

        if apply_normalization_log1p:
            if self.verbose:
                print("Applying log1p normalization...")
            X_concated = self._normalize_log1p(X_concated)

        self.X_normalized = self._split_dataframe_by_row(
            X_concated.copy(), [len(x) for x in self.X_raw]
        )

        if self.verbose:
            print("Applying PCA...")
        X_concated, pca_model = self._preprocess_pca(X_concated, pca_random_state)

        if self.verbose:
            print(
                f"\tsum of explained_variance_ratio = {sum(pca_model.explained_variance_ratio_ * 100)}"
            )

        X = self._split_dataframe_by_row(X_concated, [len(x) for x in self.X_raw])

        self.X_PCA = [df.copy() for df in X]
        self.pca_model = pca_model

        return X, pca_model

    def _apply_umap_to_concated_data(
        self,
        X_concated,
        random_state=None,
        min_dist=0.8,
    ):
        umap_model = umap.UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            random_state=random_state,
            min_dist=min_dist,
        )
        X_concated = pd.DataFrame(
            umap_model.fit_transform(X_concated.values),
            index=X_concated.index,
            columns=["UMAP1", "UMAP2"],
        )
        return X_concated, umap_model

    def apply_umap(self, random_state=None, min_dist=0.8):
        if self.X_UMAP is not None:
            return self.X_UMAP, self.umap_model

        X_concated = pd.concat(self.X_PCA)
        X_concated, umap_model = self._apply_umap_to_concated_data(
            X_concated, random_state, min_dist
        )
        X = self._split_dataframe_by_row(X_concated, [len(x) for x in self.X_raw])

        self.X_UMAP = [df.copy() for df in X]
        self.umap_model = umap_model

        return X, umap_model

    def fit_gmm(self, random_state=None):
        if self.gmm_models is not None:
            return self.gmm_models

        gmm_models = []

        for i in (
            tqdm(range(len(self.X_PCA))) if self.verbose else range(len(self.X_PCA))
        ):
            gmm_model = GaussianMixture(
                self.gmm_n_components_list[i],
                covariance_type="full",
                max_iter=2000,
                n_init=10,
                random_state=random_state,
            )
            gmm_model.fit(self.X_PCA[i].values)
            gmm_models.append(gmm_model)

        self.gmm_models = gmm_models

        return gmm_models

    def fit_predict_gmm(self, random_state=None):
        if self.gmm_models is not None and self.gmm_labels is not None:
            return self.gmm_models, self.gmm_labels

        gmm_models, gmm_labels = [], []
        for i in (
            tqdm(range(len(self.X_PCA))) if self.verbose else range(len(self.X_PCA))
        ):
            if self.gmm_models is None:
                gmm_model = GaussianMixture(
                    self.gmm_n_components_list[i],
                    covariance_type="full",
                    max_iter=2000,
                    n_init=10,
                    random_state=random_state,
                )
                gmm_labels.append(gmm_model.fit_predict(self.X_PCA[i].values))
                gmm_models.append(gmm_model)
            else:
                gmm_labels.append(self.gmm_models[i].predict(self.X_PCA[i].values))

        if self.gmm_models is None:
            self.gmm_models = gmm_models
        self.gmm_labels = gmm_labels

        return gmm_labels, gmm_models

    def predict_gmm_label(self, X_item, gmm_model):
        return gmm_model.predict(X_item.values)

    def predict_gmm_labels(self, X, gmm_models):
        gmm_labels = [
            self.predict_gmm_label(X[i], gmm_models[i]) for i in range(len(X))
        ]
        if self.gmm_labels is None:
            self.gmm_labels = gmm_labels
        return gmm_labels

    def _plot_gmm_predictions(
        self,
        X_item,
        x_range,
        y_range,
        figure_labels=None,
        gmm_labels=None,
        gmm_n_components=None,
        cmap="plasma",
    ):
        if gmm_labels is None:
            plt.scatter(X_item.values[:, 0], X_item.values[:, 1], s=0.5, alpha=0.5)
        else:
            plt.scatter(
                X_item.values[:, 0],
                X_item.values[:, 1],
                c=gmm_labels,
                alpha=0.5,
                cmap=plt.cm.get_cmap(cmap, gmm_n_components),
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
        x_range=None,
        y_range=None,
        figure_labels=None,
        figure_titles_without_gmm=None,
        figure_titles_with_gmm=None,
        plot_gmm_means=False,
        save=False,
        save_paths=None,
        cmap="plasma",
    ):
        X = self.X_PCA if mode == "pca" else self.X_UMAP

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
                for k in range(self.gmm_n_components_list[i]):
                    plt.plot(
                        gmm_model.means_[k][0],
                        gmm_model.means_[k][1],
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
                self.gmm_labels[i],
                self.gmm_n_components_list[i],
                cmap,
            )
            plt.title(figure_titles_with_gmm[i], fontsize=20)

            if save:
                plt.savefig(save_paths[i], dpi=600)
            plt.show()

    def interpolation_contour(
        self, gmm_source, gmm_target, t, x_range, y_range, cmap="rainbow"
    ):
        K0, K1 = gmm_source.means_.shape[0], gmm_target.means_.shape[0]

        solution = self.calculate_solution(gmm_source, gmm_target)
        pit = solution.reshape(K0 * K1, 1).T

        mut, St = self.calculate_mut_St(gmm_source, gmm_target, t)
        x, y = np.meshgrid(
            np.linspace(x_range[0], x_range[1], 1000),
            np.linspace(y_range[0], y_range[1], 1000),
        )
        xx = np.array([x.ravel(), y.ravel()]).T
        z = self.theoretical_density_2d(mut[:, 0:2], St[:, 0:2, 0:2], pit, xx)
        z = z.reshape(x.shape)
        max_z = np.max(z)
        min_z = np.min(z)
        img = plt.contour(x, y, z, np.linspace(min_z - 1e-9, max_z, 20), cmap=cmap)
        return img.collections

    def animate_cell_state(
        self,
        x_range=None,
        y_range=None,
        save=False,
        save_path=None,
        cmap="rainbow",
    ):
        if save and save_path is None:
            save_path = "./cell_state_video.gif"

        if x_range is None:
            x_min = min([np.min(df.iloc[:, 0].values) for df in self.X_PCA])
            x_max = max([np.max(df.iloc[:, 0].values) for df in self.X_PCA])
            x_range = (x_min - (x_max - x_min) / 10, x_max + (x_max - x_min) / 10)

        if y_range is None:
            y_min = min([np.min(df.iloc[:, 1].values) for df in self.X_PCA])
            y_max = max([np.max(df.iloc[:, 1].values) for df in self.X_PCA])
            y_range = (y_min - (y_max - y_min) / 10, y_max + (y_max - y_min) / 10)

        ims = []

        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(len(self.day_names) - 1):
            t = np.linspace(0, 1, 11)
            for j in tqdm(range(11)) if self.verbose else range(11):
                im = self.interpolation_contour(
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

        anim = animation.ArtistAnimation(fig, ims, interval=100)
        plt.close()
        display(HTML(anim.to_jshtml()))

        if save:
            anim.save(save_path, writer="pillow")

    def plot_cell_state_graph(
        self,
        cluster_names,
        tf_gene_names,
        mode="pca",
        tf_gene_pick_num=5,
        thresh=0.05,
        save=False,
        save_path=None,
    ):
        if save and save_path is None:
            save_path = "./cell_state_graph.png"

        gmm_means = self.get_gmm_means()

        gmm_means_flattened = np.array(list(itertools.chain.from_iterable(gmm_means)))
        if mode == "umap":
            gmm_means_flattened = self.umap_model.transform(gmm_means_flattened)

        cell_state_graph = self._make_cell_state_graph(cluster_names, thresh)
        G = nx.from_pandas_edgelist(
            cell_state_graph,
            source="source",
            target="target",
            edge_attr=["edge_weights", "edge_colors"],
            create_using=nx.DiGraph,
        )
        node_weights_and_pos = pd.DataFrame(
            self._get_gmm_node_weights_flattened(),
            index=list(itertools.chain.from_iterable(cluster_names)),
            columns=["node_weights"],
        )
        node_weights_and_pos["xpos"] = gmm_means_flattened.T[0]
        node_weights_and_pos["ypos"] = gmm_means_flattened.T[1]
        node_weights_and_pos["node_days"] = LabelEncoder().fit_transform(
            self._get_day_names_of_each_node()
        )
        pos = {}
        for row in node_weights_and_pos.itertuples():
            G.add_node(row.Index, weight=row.node_weights)
            G.add_node(row.Index, day=row.node_days)
            pos[row.Index] = (row.xpos, row.ypos)

        mean_gene_values_per_cluster = self.get_gmm_mean_gene_values_per_cluster(
            gmm_means,
            list(itertools.chain.from_iterable(cluster_names)),
        )
        mean_tf_gene_values_per_cluster = mean_gene_values_per_cluster.loc[
            :, mean_gene_values_per_cluster.columns.isin(tf_gene_names)
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
            mean_tf_gene_values_per_cluster, cell_state_graph, num=tf_gene_pick_num
        )
        tf_down_genes = self._get_down_regulated_genes(
            mean_tf_gene_values_per_cluster, cell_state_graph, num=tf_gene_pick_num
        )
        tf_up_genes.columns += 1
        tf_down_genes.columns += 1

        self._plot_cell_state_graph(
            G,
            pos,
            nodes_up_gene=tf_nlargest,
            nodes_down_gene=tf_nsmallest,
            edges_up_gene=tf_up_genes,
            edges_down_gene=tf_down_genes,
            save=save,
            save_path=save_path,
        )

    def _get_gmm_node_weights_flattened(self):
        node_weights = [
            self.gmm_models[i].weights_ for i in range(len(self.gmm_models))
        ]
        node_weights = itertools.chain.from_iterable(node_weights)
        return node_weights

    def _get_day_names_of_each_node(self):
        day_names_of_each_node = []
        for i, gmm_n_components in enumerate(self.gmm_n_components_list):
            day_names_of_each_node += [self.day_names[i]] * gmm_n_components
        return day_names_of_each_node

    def _get_nlargest_gene_indices(self, row, num=10):
        nlargest = row.nlargest(num)
        return nlargest.index

    def _get_nsmallest_gene_indices(self, row, num=10):
        nsmallest = row.nsmallest(num)
        return nsmallest.index

    def _make_cell_state_graph(self, cluster_names, thresh):
        node_source_target_combinations, edge_colors_based_on_source = [], []
        for i in range(len(self.gmm_n_components_list) - 1):
            current_combinations = [
                x for x in itertools.product(cluster_names[i], cluster_names[i + 1])
            ]
            node_source_target_combinations += current_combinations
            edge_colors_based_on_source += [i for j in range(len(current_combinations))]
        cell_state_graph = pd.DataFrame(
            node_source_target_combinations, columns=["source", "target"]
        )
        cell_state_graph["edge_colors"] = edge_colors_based_on_source
        cell_state_graph["edge_weights"] = list(
            itertools.chain.from_iterable(
                list(
                    itertools.chain.from_iterable(
                        self.calculate_normalized_solutions(self.gmm_models)
                    )
                )
            )
        )
        cell_state_graph = cell_state_graph[cell_state_graph["edge_weights"] > thresh]

        return cell_state_graph

    def _get_fold_change(self, gene_values, source, target):
        fold_change = pd.Series(
            gene_values.T[target] - gene_values.T[source], index=gene_values.T.index
        )
        fold_change = fold_change.sort_values(ascending=False)
        return fold_change

    def _get_up_regulated_genes(self, gene_values, cell_state_graph, num=10):
        df_upgenes = pd.DataFrame([])
        for i in range(len(cell_state_graph)):
            s1 = cell_state_graph["source"].iloc[i]
            s2 = cell_state_graph["target"].iloc[i]
            fold_change = self._get_fold_change(gene_values, s1, s2)
            upgenes = pd.Series(
                self._get_nlargest_gene_indices(fold_change, num=num).values
            )
            df_upgenes = df_upgenes.append(upgenes, ignore_index=True)
        df_upgenes.index = cell_state_graph["source"] + cell_state_graph["target"]
        df_upgenes.columns = df_upgenes.columns
        return df_upgenes

    def _get_down_regulated_genes(self, gene_values, cell_state_graph, num=10):
        df_downgenes = pd.DataFrame([])
        for i in range(len(cell_state_graph)):
            s1 = cell_state_graph["source"].iloc[i]
            s2 = cell_state_graph["target"].iloc[i]
            fold_change = self._get_fold_change(gene_values, s1, s2)
            downgenes = pd.Series(
                self._get_nsmallest_gene_indices(fold_change, num=num).values
            )
            df_downgenes = df_downgenes.append(downgenes, ignore_index=True)
        df_downgenes.index = cell_state_graph["source"] + cell_state_graph["target"]
        df_downgenes.columns = df_downgenes.columns
        return df_downgenes

    def _plot_cell_state_graph(
        self,
        G,
        pos,
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
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            tail_list.append(pos[edge[0]])
            head_list.append(pos[edge[1]])
            weight = G.edges[edge]["edge_weights"] * 25
            color = colors[G.edges[edge]["edge_colors"] + 1]

            color_list.append(f"rgb({color[0]},{color[1]},{color[2]})")

            edge_trace = go.Scatter(
                x=tuple([x0, x1, None]),
                y=tuple([y0, y1, None]),
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
                "color": [w["edge_colors"] + 1 for w in G.edges.values()],
            },
            opacity=0,
        )

        for edge in G.edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            from_to = str(edge[0]) + str(edge[1])
            hovertext1 = edges_up_gene.T[from_to].values
            hovertext2 = edges_down_gene.T[from_to].values
            hovertext = (
                "up_gene: " + hovertext1 + " " + "down_gene: " + hovertext2 + "<br>"
            )
            middle_hover_trace["x"] += tuple([(x0 + x1) / 2])
            middle_hover_trace["y"] += tuple([(y0 + y1) / 2])
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
                "color": [w["day"] + 1 for w in G.nodes.values()],
            },
            opacity=0,
        )

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            hovertext1 = nodes_up_gene.T[node].values
            hovertext2 = nodes_down_gene.T[node].values
            hovertext = (
                "largest_gene: "
                + hovertext1
                + " "
                + "smallest_gene: "
                + hovertext2
                + "<br>"
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

        node_trace.marker.color = [w["day"] for w in G.nodes.values()]
        node_trace.marker.size = [w["weight"] * 140 for w in G.nodes.values()]
        trace_recode.append(node_trace)

        fig = go.Figure(
            data=trace_recode, layout=go.Layout(showlegend=False, hovermode="closest")
        )

        fig.update_layout(annotations=arrows)

        fig.update_layout(width=1000, height=800, title="Cell state graph")
        fig.show()

        if save:
            fig.write_image(save_path)

    def plot_fold_change(
        self,
        cluster_names,
        tf_gene_names,
        cluster1,
        cluster2,
        threshold=1.0,
        save=False,
        save_path=None,
    ):
        if save and save_path is None:
            save_path = "./fold_change.png"

        genes = self.get_gmm_mean_gene_values_per_cluster(
            self.get_gmm_means(),
            cluster_names=list(itertools.chain.from_iterable(cluster_names)),
        )
        genes = genes.loc[:, genes.columns.isin(tf_gene_names)]
        genes = genes.T
        genes_fold_change = pd.DataFrame(index=genes.index)
        genes_fold_change[cluster1] = genes[cluster1]
        genes_fold_change[cluster2] = genes[cluster2]

        fig = go.Figure()
        gene_exp1 = genes_fold_change[cluster1]
        gene_exp2 = genes_fold_change[cluster2]
        absFC = (gene_exp2 - gene_exp1).abs()
        genes_fold_change = genes_fold_change.loc[
            genes_fold_change.index.isin(absFC[absFC.values > threshold].index), :
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
        tf_gene_names,
        pathway_names,
        threshold=1.0,
        save=False,
        save_path=None,
    ):
        if save and save_path is None:
            save_path = "./pathway_mean_var.png"

        genes = self.get_gmm_mean_gene_values_per_cluster(
            self.get_gmm_means(),
            cluster_names=list(itertools.chain.from_iterable(cluster_names)),
        )
        genes = genes.loc[:, genes.columns.isin(tf_gene_names)]

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
        tf_gene_names,
        pathway_names,
        selected_genes,
        save=False,
        save_path=None,
    ):
        if save and save_path is None:
            save_path = "./pathway_gene_expressions.png"

        genes = self.get_gmm_mean_gene_values_per_cluster(
            self.get_gmm_means(),
            cluster_names=list(itertools.chain.from_iterable(cluster_names)),
        )
        genes = genes.loc[:, genes.columns.isin(tf_gene_names)]

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
        if save and save_path is None:
            save_path = "./pathway_single_gene_2d.png"

        X_concated = pd.concat(self.X_PCA if mode == "pca" else self.X_UMAP)
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
        self, gene_name, mode="pca", col=None, save=False, save_path=None
    ):
        if save and save_path is None:
            save_path = "./pathway_single_gene_3d.html"

        X_concated = pd.concat(self.X_PCA if mode == "pca" else self.X_UMAP)
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
        self, gmm_source, gmm_target, t, columns=None, n_samples=2000
    ):
        d = gmm_source.means_.shape[1]
        K0, K1 = gmm_source.means_.shape[0], gmm_target.means_.shape[0]

        mu0, mu1 = gmm_source.means_, gmm_target.means_
        S0, S1 = gmm_source.covariances_, gmm_target.covariances_

        pi0, pi1 = gmm_source.weights_, gmm_target.weights_

        solution = self.EGOT(
            np.ravel(pi0),
            np.ravel(pi1),
            mu0.reshape(K0, d),
            mu1.reshape(K1, d),
            S0.reshape(K0, d, d),
            S1.reshape(K1, d, d),
        )
        pit = solution.reshape(K0 * K1, 1).T

        mut, St = self.calculate_mut_St(gmm_source, gmm_target, t)

        K = mut.shape[0]
        pit = pit.reshape(1, K)
        means = mut
        covariances = St
        weights = pit[0, :]
        rng = check_random_state(0)
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
        X_interpolation = self.make_interpolation_data(
            self.gmm_models[interpolate_index - 1],
            self.gmm_models[interpolate_index + 1],
            t,
            self.X_PCA[0].columns,
        )
        X_true = self.X_PCA[interpolate_index]
        X_source = self.X_PCA[interpolate_index - 1]
        X_target = self.X_PCA[interpolate_index + 1]

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

    def animate_interpolation(
        self,
        target_gene_name,
        mode="pca",
        interpolate_interval=10,
        n_samples=5000,
        x_range=None,
        y_range=None,
        c_range=None,
        x_label=None,
        y_label=None,
        cmap="rainbow",
        save=False,
        save_path=None,
    ):
        X = self.X_PCA if mode == "pca" else self.X_UMAP
        gene_expression_level = pd.concat(self.X_normalized)[target_gene_name]

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
            for j in (
                tqdm(range(interpolate_interval))
                if self.verbose
                else range(interpolate_interval)
            ):
                X_interpolation = self.make_interpolation_data(
                    self.gmm_models[i],
                    self.gmm_models[i + 1],
                    t[j],
                    columns=self.X_PCA[0].columns,
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
                        self.umap_model.transform(X_interpolation)
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

        anim_gene = animation.ArtistAnimation(fig, ims, interval=100)
        plt.close()
        display(HTML(anim_gene.to_jshtml()))

        if save:
            anim_gene.save(save_path, writer="pillow")

    def prepare_cell_velocity(self, mode="pca"):
        x_velocity = []
        y_velocity = []
        speed = []

        if self.solutions is None:
            self.solutions = self.calculate_solutions(self.gmm_models)

        for i in (
            tqdm(range(len(self.gmm_models) - 1))
            if self.verbose
            else range(len(self.gmm_models) - 1)
        ):
            gmm_source = self.gmm_models[i]
            gmm_target = self.gmm_models[i + 1]

            velocity = self.compute_cell_velocity(
                gmm_source, gmm_target, self.X_PCA[i], self.solutions[i]
            )
            if mode == "umap":
                velocity = self.umap_model.transform(velocity)
            U = velocity[:, 0]
            V = velocity[:, 1]

            x_velocity = np.append(x_velocity, U)
            y_velocity = np.append(y_velocity, V)
            speed = np.append(speed, np.sqrt(U**2 + V**2))

        return x_velocity, y_velocity, speed

    def get_gaussian_map(self, m0, m1, Sigma0, Sigma1, x):
        d = Sigma0.shape[0]
        m0 = m0.reshape(1, d)
        m1 = m1.reshape(1, d)
        Sigma0 = Sigma0.reshape(d, d)
        Sigma1 = Sigma1.reshape(d, d)
        Sigma = np.linalg.pinv(Sigma0) @ spl.sqrtm(Sigma0 @ Sigma1)
        Tx = m1 + (x - m0) @ Sigma
        return Tx

    def compute_cell_velocity(self, gmm_source, gmm_target, X_item, solution):
        d = gmm_source.means_.shape[1]
        K0, K1 = gmm_source.means_.shape[0], gmm_target.means_.shape[0]

        mu0, mu1 = gmm_source.means_, gmm_target.means_
        S0, S1 = gmm_source.covariances_, gmm_target.covariances_

        n = X_item.shape[0]
        T = np.zeros((K0, K1, d, n))
        barycentric_projection_map = np.zeros((d, n))
        Nj = np.zeros((n, K0))

        for k in range(K0):
            for l in range(K1):
                T[k, l, :, :] = self.get_gaussian_map(
                    mu0[k, :], mu1[l, :], S0[k, :, :], S1[l, :, :], X_item.values
                ).T
        for j in range(K0):
            logprob = gmm_source.score_samples(X_item.values)
            Nj[:, j] = np.exp(
                np.log(
                    multivariate_normal.pdf(
                        X_item.values, mean=mu0[j, :], cov=S0[j, :, :]
                    )
                )
                - logprob
            )
        for k in range(K0):
            for l in range(K1):
                barycentric_projection_map += (
                    solution[k, l] * Nj[:, k].T * T[k, l, :, :]
                )
        w = barycentric_projection_map.T
        velo = w - X_item.values
        return velo

    def plot_cell_velocity(
        self,
        x_velocity,
        y_velocity,
        speed,
        cmap="rainbow",
        mode="PCA",
        save=False,
        save_path=None,
    ):
        if save and save_path is None:
            save_path = "./cell_velocity.png"

        scale = 1 if mode == "pca" else 2.5
        margin = 5 if mode == "pca" else 1

        X_concated = pd.concat(self.X_PCA[:-1] if mode == "pca" else self.X_UMAP[:-1])

        x_coordinate = X_concated.iloc[:, 0]
        y_coordinate = X_concated.iloc[:, 1]

        plt.figure(figsize=(10, 8))
        plt.quiver(
            x_coordinate,
            y_coordinate,
            x_velocity / speed,
            y_velocity / speed,
            speed,
            cmap=cmap,
            scale=scale,
            scale_units="xy",
        )
        plt.xlim(np.min(x_coordinate) - margin, np.max(x_coordinate) + margin)
        plt.ylim(np.min(y_coordinate) - margin, np.max(y_coordinate) + margin)
        plt.xlabel(X_concated.columns[0])
        plt.ylabel(X_concated.columns[1])

        plt.colorbar()

        if save:
            plt.savefig(save_path)

    def make_GRN_graph(self, df, threshold=0.1):
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

    def plot_interpolation_of_cell_velocity(
        self,
        x_velocity,
        y_velocity,
        color_streams=False,
        color_points="gmm",
        cluster_names=None,
        mode="pca",
        x_range=None,
        y_range=None,
        cmap="rainbow",
        linspace_num=300,
        save=False,
        save_path=None,
    ):
        if save and save_path is None:
            save_path = "./interpolation_of_cell_velocity_gmm_clusters.png"

        X = self.X_PCA if mode == "pca" else self.X_UMAP

        colors = []
        if color_points == "gmm":
            label_sum = 0
            for i in range(len(self.gmm_labels)):
                colors += [label + label_sum for label in self.gmm_labels[i]]
                label_sum += self.gmm_n_components_list[i]
        elif color_points == "day":
            for i in range(len(X)):
                colors += [1] * len(X)

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
                cmap=plt.cm.get_cmap(cmap, len(set(colors))),
                s=20,
                alpha=0.5,
            )
            if color_points == "gmm" and cluster_names is not None:
                plt.legend(
                    handles=scatter.legend_elements(num=len(set(colors)))[0],
                    labels=cluster_names,
                )
            if color_points == "day":
                plt.legend(
                    handles=scatter.legend_elements(num=len(set(colors)))[0],
                    labels=self.day_names,
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

        if save:
            plt.savefig(save_path)

    def calculate_GRNs(
        self,
    ):
        GRNs, ridgeCVs = [], []

        if self.solutions is None:
            self.solutions = self.calculate_solutions(self.gmm_models)

        for i in (
            tqdm(range(len(self.gmm_models) - 1))
            if self.verbose
            else range(len(self.gmm_models) - 1)
        ):
            gmm_source = self.gmm_models[i]
            gmm_target = self.gmm_models[i + 1]

            velo = self.compute_cell_velocity(
                gmm_source, gmm_target, self.X_PCA[i], self.solutions[i]
            )

            df_X_inverse = pd.DataFrame(
                self.pca_model.inverse_transform(self.X_PCA[i].values),
                columns=self.gene_names,
            )

            df_velo_inverse = pd.DataFrame(
                self.pca_model.inverse_transform(velo), columns=self.gene_names
            )

            alphas_cv = np.logspace(-2, 2, num=20)
            ridgeCV = linear_model.RidgeCV(alphas=alphas_cv, cv=3, fit_intercept=False)
            ridgeCV.fit(df_X_inverse, df_velo_inverse)
            ridgeCVs.append(ridgeCV)

            GRN = linear_model.Ridge(alpha=ridgeCV.alpha_, fit_intercept=False)
            GRN.fit(df_X_inverse, df_velo_inverse)
            df_GRN = pd.DataFrame(
                GRN.coef_, index=self.gene_names, columns=self.gene_names
            )

            GRNs.append(df_GRN)

        return GRNs, ridgeCVs

    def plot_GRN_graph(
        self,
        GRNs,
        ridgeCVs,
        selected_genes,
        thresh=0.01,
        save=False,
        save_paths=None,
        save_format="png",
    ):
        if save and save_paths is None:
            save_paths = [f"./GRN_graph_{i + 1}.png" for i in range(len(GRNs))]
        for i, GRN in enumerate(GRNs):
            if self.verbose:
                print(f"alpha = {ridgeCVs[i].alpha_}")
            GRNgraph = self.make_GRN_graph(
                GRN[selected_genes].loc[selected_genes], threshold=thresh
            )
            display(Image(GRNgraph.create(format=save_format)))
            if save:
                GRNgraph.write(save_paths[i], format=save_format)

    def calculate_waddington_potential(self, n_neighbors=100):
        if self.solutions is None:
            self.solutions = self.calculate_solutions(self.gmm_models)

        F_all = []

        for i in (
            tqdm(range(len(self.X_PCA) - 1))
            if self.verbose
            else range(len(self.X_PCA) - 1)
        ):
            solution = self.solutions[i]

            gmm_source, gmm_target = self.gmm_models[i], self.gmm_models[i + 1]

            d = gmm_source.means_.shape[1]
            K0, K1 = gmm_source.means_.shape[0], gmm_target.means_.shape[0]

            mu0, mu1 = gmm_source.means_, gmm_target.means_
            S0, S1 = gmm_source.covariances_, gmm_target.covariances_

            pi0 = gmm_source.weights_

            B = 0
            F = 0
            for j in range(K0):
                B = B + np.nan_to_num(
                    np.dot(
                        np.linalg.pinv(S0)[j, :, :],
                        (self.X_PCA[i].values - mu0[j, :]).T,
                    )
                    * pi0[j]
                    * multivariate_normal.pdf(
                        self.X_PCA[i].values, mean=mu0[j, :], cov=S0[j, :, :]
                    )
                    / self.theoretical_density_2d(mu0, S0, pi0, self.X_PCA[i].values)
                )
            for j in range(K0):
                A = np.dot(
                    np.linalg.pinv(S0)[j, :, :], (self.X_PCA[i].values - mu0[j, :]).T
                )
                I1 = -A + B
                for k in range(K1):
                    P = (
                        solution[j, k]
                        * multivariate_normal.pdf(
                            self.X_PCA[i].values, mean=mu0[j, :], cov=S0[j, :, :]
                        )
                        / self.theoretical_density_2d(
                            mu0, S0, pi0, self.X_PCA[i].values
                        ).T
                    )
                    P = np.nan_to_num(P)
                    Tmap = self.get_gaussian_map(
                        mu0[j, :],
                        mu1[k, :],
                        S0[j, :, :],
                        S1[k, :, :],
                        self.X_PCA[i].values,
                    ).T
                    Tmap = Tmap.real
                    I2 = np.nan_to_num(
                        np.trace(
                            np.linalg.pinv(S0[j, :, :])
                            @ spl.sqrtm(S0[j, :, :] @ S1[k, :, :])
                        )
                    )
                    I2 = I2.real
                    F = F + np.sum(np.dot(I1, Tmap.T)) * P + I2 * P
            F = F - d
            F_all = np.append(F_all, F)

        if self.verbose:
            print("Applying knn ...")
        knn = kneighbors_graph(
            X=pd.concat(self.X_PCA[:-1]).iloc[:, :2].values,
            n_neighbors=n_neighbors,
            mode="distance",
            metric="euclidean",
        )

        if self.verbose:
            print("Computing kernel ...")
        sim = csr_matrix(knn.shape)

        nonzero = knn.nonzero()
        sig = 10
        sim[nonzero] = np.exp(-np.array(knn[nonzero]) ** 2 / sig**2)
        sim = (sim + sim.T) / 2
        deg = sim.sum(axis=1)
        n = pd.concat(self.X_PCA[:-1]).iloc[:, :2].values.shape[0]
        dia = np.diag(
            np.array(deg).reshape(
                n,
            )
        )
        lap = dia - sim
        lap = csc_matrix(np.array(lap))
        Wpotential, *_ = linalg.lsqr(lap, F_all)

        Wpotential = zscore(Wpotential)

        return Wpotential, F_all

    def plot_waddington_potential(
        self,
        Wpotential,
        gene_name=None,
        save=False,
        save_path=None,
    ):
        if save and save_path is None:
            save_path = "./waddington_potential.html"

        X_concated = pd.concat(self.X_PCA[:-1])
        gene_expression_level = None
        if gene_name:
            gene_expression_level = pd.concat(self.X_normalized)[: len(Wpotential)][
                gene_name
            ]
        plot_data = pd.DataFrame(index=X_concated.index)
        plot_data["x"] = X_concated.iloc[:, 0]
        plot_data["y"] = X_concated.iloc[:, 1]
        plot_data["z"] = Wpotential
        fig = px.scatter_3d(
            plot_data,
            x="x",
            y="y",
            z="z",
            color=gene_expression_level,
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
        Wpotential,
        gene_name=None,
        save=False,
        save_path=None,
    ):
        if save and save_path is None:
            save_path = "./wadding_potential_surface.html"

        X_concated = pd.concat(self.X_PCA[:-1])
        x, y = X_concated.iloc[:, 0], X_concated.iloc[:, 1]

        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)

        X, Y = np.meshgrid(xi, yi)

        Z = interpolate.griddata((x, y), Wpotential, (X, Y), method="cubic")

        fig = go.Figure(
            go.Surface(
                x=xi,
                y=yi,
                z=Z,
            )
        )

        fig.update_scenes(
            xaxis_title_text=X_concated.columns[0],
            yaxis_title_text=X_concated.columns[1],
            zaxis_title_text="Waddington Potential",
        )

        fig.update_layout(width=1000, height=800)

        if gene_name is None:
            fig.update_layout(title="Waddington potential")
        else:
            fig.update_layout(title=f"Waddington potential, gene = {gene_name}")
        fig.show()

        if save:
            fig.write_html(save_path)

    def GaussianW(self, m0, m1, Sigma0, Sigma1):
        Sigma00 = spl.sqrtm(Sigma0)
        Sigma010 = spl.sqrtm(Sigma00 @ Sigma1 @ Sigma00)
        d = np.linalg.norm(m0 - m1) ** 2 + np.trace(Sigma0 + Sigma1 - 2 * Sigma010)
        return d

    def EGOT(self, pi0, pi1, mu0, mu1, S0, S1):
        K0 = mu0.shape[0]
        K1 = mu1.shape[0]
        d = mu0.shape[1]
        S0 = S0.reshape(K0, d, d)
        S1 = S1.reshape(K1, d, d)
        M = np.zeros((K0, K1))
        for k in range(K0):
            for l in range(K1):
                M[k, l] = self.GaussianW(mu0[k, :], mu1[l, :], S0[k, :, :], S1[l, :, :])
        solution = ot.sinkhorn(
            pi0,
            pi1,
            M / M.max(),
            0.01,
            method="sinkhorn_epsilon_scaling",
            numItermax=10000000000,
            tau=1e8,
            stopThr=1e-9,
        )
        return solution

    def calculate_solution(self, gmm_source, gmm_target):
        pi0, pi1 = gmm_source.weights_, gmm_target.weights_
        mu0, mu1 = gmm_source.means_, gmm_target.means_
        S0, S1 = gmm_source.covariances_, gmm_target.covariances_

        solution = self.EGOT(
            pi0,
            pi1,
            mu0,
            mu1,
            S0,
            S1,
        )
        return solution

    def calculate_solutions(self, gmm_models):
        solutions = []
        for i in range(len(gmm_models) - 1):
            solutions.append(self.calculate_solution(gmm_models[i], gmm_models[i + 1]))
        return solutions

    def calculate_normalized_solutions(self, gmm_models):
        solutions_normalized = []
        for i in range(len(gmm_models) - 1):
            solution = self.calculate_solution(gmm_models[i], gmm_models[i + 1])
            solutions_normalized.append((solution.T / gmm_models[i].weights_).T)
        return solutions_normalized

    def theoretical_density_2d(self, mu, Sigma, alpha, x):
        K = mu.shape[0]
        alpha = alpha.reshape(1, K)
        y = 0
        for j in range(K):
            y += alpha[0, j] * multivariate_normal.pdf(
                x, mean=mu[j, :], cov=Sigma[j, :, :]
            )
        return y

    def calculate_mut_St(self, gmm_source, gmm_target, t):
        d = gmm_source.means_.shape[1]
        K0, K1 = gmm_source.means_.shape[0], gmm_target.means_.shape[0]

        mu0, mu1 = gmm_source.means_, gmm_target.means_
        S0, S1 = gmm_source.covariances_, gmm_target.covariances_

        mut = np.zeros((K0 * K1, d))
        St = np.zeros((K0 * K1, d, d))
        for k in range(K0):
            for l in range(K1):
                mut[k * K1 + l, :] = (1 - t) * mu0[k, :] + t * mu1[l, :]
                Sigma1demi = spl.sqrtm(S1[l, :, :])
                C = (
                    Sigma1demi
                    @ spl.inv(spl.sqrtm(Sigma1demi @ S0[k, :, :] @ Sigma1demi))
                    @ Sigma1demi
                )
                St[k * K1 + l, :, :] = (
                    ((1 - t) * np.eye(d) + t * C)
                    @ S0[k, :, :]
                    @ ((1 - t) * np.eye(d) + t * C)
                )

        return mut, St

    def generate_cluster_names_with_day(self, cluster_names=None):
        if cluster_names is None:
            cluster_names = []
            for i in range(len(self.gmm_n_components_list)):
                cluster_names.append(
                    [f"{j}" for j in range(self.gmm_n_components_list[i])]
                )

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

    def get_gmm_mean_gene_values_per_cluster(self, gmm_means, cluster_names=None):
        gmm_mean_gene_values_per_cluster = self._get_gmm_mean_gene_values_per_cluster(
            gmm_means, cluster_names
        )
        gmm_mean_gene_values_per_cluster = gmm_mean_gene_values_per_cluster.where(
            gmm_mean_gene_values_per_cluster > 0, 0
        )
        return gmm_mean_gene_values_per_cluster