import numpy as np
import matplotlib.pyplot as plt
import time
import os

import warnings

from matplotlib import MatplotlibDeprecationWarning

warnings.simplefilter("ignore", MatplotlibDeprecationWarning)  # 或者 MatplotlibDeprecationWarning

class GraphPlot:
    def __init__(self):
        self.No = 0
        self.cmap = plt.get_cmap('hsv')

    def draw_graph(self, graph, draw=True, title=None):
        assert graph.shape[-1] == 2, "only support 2d data"
        if len(graph.shape) == 3 and graph.shape[0] == 1:
            graph = graph.squeeze(0)
        assert len(graph.shape) == 2 , "only support one group of data"
        fig, axs = plt.subplots(1, 1)
        axs.plot(graph[:, 0], graph[:, 1], "b.")
        self.plot_format(axs, title=title)
        if draw:
            fig.show()
        return fig

    def draw_route(self, graph, route_index, draw=True, title=None, one_first = False):
        """
        绘制路线, 城市数量N， 路线数量M, 地图维度2， 轨迹长度S(不定长)
        :param title:
        :param graph: [N,2]
        :param route_index: List[List[int]] [M_i,S_i] (i = 0,...,M-1)
        :return: fig
        """
        assert graph.shape[-1] == 2, "only support 2d data"
        if len(graph.shape) == 3 and graph.shape[0] == 1:
            graph = graph.squeeze(0)
        assert len(graph.shape) == 2 , "only support one group of data"
        fig, axs = plt.subplots(1, 1)
        # axs.plot(graph[:, 0], graph[:, 1], ".", )

        M = len(route_index)
        routes = [[graph[j-1 if one_first else j] for j in route_index[i]] for i in range(M)]
        cnt = 0 # 路线长度
        for i in range(M):
            if len(routes[i]) == 0:
                continue
            cnt += 1
            route = np.array(routes[i])
            axs.plot(route[:, 0], route[:, 1],"o-", color=self.cmap(1.0 * cnt / M))
        self.plot_format(axs, title=title)
        if draw:
            fig.show()
        return fig

    def combine_figs(self, figs):
        # Calculate grid dimensions based on number of figures
        row = int(np.sqrt(len(figs)))
        col = len(figs) // row + (1 if len(figs) % row else 0)

        # Create a new combined figure with appropriate dimensions
        fig_combined, axs = plt.subplots(row, col, figsize=(5 * col, 5 * row))

        # Convert axes to flat list for easier iteration
        if isinstance(axs, np.ndarray):
            axs = axs.flatten().tolist()
        elif not isinstance(axs, list):
            axs = [axs]  # Handle case with only one subplot

        # Copy content from each source figure to the corresponding subplot
        for i, (source_fig, target_ax) in enumerate(zip(figs, axs)):
            source_ax = source_fig.axes[0]  # Get the first Axes from the original figure

            # Copy all lines (line plots)
            for line in source_ax.get_lines():
                target_ax.plot(line.get_xdata(), line.get_ydata(),
                               color=line.get_color(),
                               linestyle=line.get_linestyle(),
                               marker=line.get_marker(),
                               markersize=line.get_markersize(),
                               label=line.get_label())

            # Copy scatter plots
            for collection in source_ax.collections:
                if isinstance(collection, plt.matplotlib.collections.PathCollection):  # Scatter plots
                    # Extract scatter plot properties
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        x = offsets[:, 0]
                        y = offsets[:, 1]
                        sizes = collection.get_sizes()
                        colors = collection.get_facecolors()
                        size = sizes[0] if len(sizes) == 1 else sizes
                        color = colors[0] if len(colors) == 1 else colors

                        # Recreate scatter plot
                        target_ax.scatter(x, y, s=size, c=color)

            # # Copy axis properties
            target_ax.set_title(source_ax.get_title())
            # target_ax.set_xlabel(source_ax.get_xlabel())
            # target_ax.set_ylabel(source_ax.get_ylabel())
            #
            # # Copy axis limits if they were explicitly set
            # target_ax.set_xlim(source_ax.get_xlim())
            # target_ax.set_ylim(source_ax.get_ylim())
            #
            # # Copy grid settings
            # target_ax.grid(source_ax.get_xgridlines() and source_ax.get_ygridlines())

        # Hide any unused subplots
        for j in range(len(figs), len(axs)):
            axs[j].axis('off')

        # Adjust layout for better spacing
        fig_combined.tight_layout()

        return fig_combined

    def plot_format(self, ax, title=None):
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title)

    def save_svg(self, fig=None, dir=None, No=None, type="route"):
        if fig is None:
            print("fig is None")
            return False

        if type not in ["route", "graph"]:
            print("save fig type error")
            return False

        file_dir = dir if dir is not None else "./graph"
        if not os.path.exists(file_dir):
            os.makedirs(dir)
        local_time = time.localtime()
        time_str = time.strftime("%y-%m-%d_%H:%M:%S", local_time)

        fig.savefig(f"{file_dir}/{No}_{type}_{time_str}.svg")

        return True


if __name__ == "__main__":
    from envs.GraphGenerator import GraphGenerator

    DG = GraphGenerator()
    data = DG.generate()
    DP = GraphPlot()
    ori_fig = DP.draw_graph(data[0],title="graph")
    idx = [
        [0, 1, 2, 0],
        [0, 4, 0],
        [0, 3, 0]
    ]
    fig = DP.draw_route(data[1], idx, title="route")
    DP.save_svg(fig=fig, dir="./graph", No=1, type="route")
    DP.save_svg(fig=ori_fig, type="graph", No=1)
