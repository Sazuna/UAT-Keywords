"""
Compute coherence between output nodes.
"""
import math
from typing import List
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from src.utils.config import ADS_CORPUS_DIR
from src.utils.corpus_loader import Reader
from src.utils.util import get_depth, uat_json

class Coherence():

    def __init__(self,
                 data: List[List]):
        """
        Args:
            data: a list containing lists of elements that occur together in the training dataset.
        """

        self.node_count = Counter()
        self.pair_count = Counter()

        for sample in data:
            unique_nodes = set(sample)
            
            for node in unique_nodes:
                self.node_count[node] += 1
            
            for i, j in combinations(sorted(unique_nodes), 2):
                self.pair_count[(i, j)] += 1

        # samples count (total of pair counts)
        self.N = sum(self.pair_count.values())

        # Plot the nodes distribution

        """
        plt.style.use('_mpl-gallery')
        # make data:
        x = 1 + np.arange(len(self.node_count))
        y = sorted(self.node_count.values(), reverse=True)

        # plot
        fig, ax = plt.subplots()

        ax.bar(x, y, width=1, edgecolor="blue", linewidth=0.7)

        #ax.set(xlim=(0, 2411), xticks=np.arange(1, 2411),
        #       ylim=(0, 4000), yticks=np.arange(1, 4000))


        y_max = max(y)
        y_ticks = np.arange(0, y_max + 500, 500)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        
        ax.set_xlabel('Node Index (ranked by frequency)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Node Frequencies', fontsize=14)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig("nodes_distrib.jpg")
        """
        # node (uri), count, depth
        with open("node_count_depth.csv", "w") as file:
            res = ""
            for node in uat_json:
            # for node, count in self.node_count.items():
                count = self.node_count[node]
                depth = get_depth(node)
                if depth < 0:
                    continue
                res += str(node) + "," + str(count) + "," + str(depth) + "\n"
            file.write(res)

        # Plot the nodes distribution
        plt.style.use('_mpl-gallery')
        # make data:
        x = 1 + np.arange(len(self.node_count))
        y = sorted(self.node_count.values(), reverse=True)
    
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(x, y, width=0.8, edgecolor="blue", linewidth=0.5, color='steelblue')

        y_max = max(y)
        y_ticks = np.arange(0, y_max + 500, 500)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)

        x_max = len(self.node_count)
        x_ticks = np.arange(0, x_max + 500, 500)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)

        ax.set_xlim(0, x_max + 10)
        ax.set_ylim(0, y_max * 1.05)
        
        ax.set_xlabel('Node Index (ranked by frequency)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Node Frequencies', fontsize=14)

        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig("nodes_distrib.jpg", dpi=150)

        # Plot the node distribution by depth
        node_depths = []
        node_depths_total = []
        for node, count in self.node_count.items():
            # Add node_count times the depth to node_depths
            depth = get_depth(node)
            if depth == -1:
                continue # Deprecated
            node_depths.extend([depth] * count)
            node_depths_total.append(1)
        c = Counter(node_depths)
        # Plot distribution: count (y) by depth (x)
        depths = sorted(c.keys())
        counts = [c[d] for d in depths]

        # Divided by count of nodes by depth in the UAT itself to see which levels are over-represented
        c2 = Counter(node_depths_total)
        c_mean = c.copy()
        counts2 = []
        for depth in depths:
            count = c[depth]
            c_mean[depth] = c[depth] / count

        depths_mean = sorted(c_mean.keys())
        counts_mean = [c_mean[d] for d in depths_mean]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(depths, counts, width=0.8, edgecolor="black", linewidth=0.5, color="darkorange")

        # X (depth)
        ax.set_xticks(depths)
        ax.set_xlim(min(depths) - 1, max(depths) + 1)

        # Y (count)
        y_max = max(counts)
        y_ticks = np.arange(0, y_max + 500, 500)
        ax.set_yticks(y_ticks)
        ax.set_xlabel('Depth', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Node Depths', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig("depth_distribution.jpg", dpi=150)

        node_depths = []
        node_depths_total = []

        for node, count in self.node_count.items():
            depth = get_depth(node)
            if depth == -1:
                continue
            node_depths.extend([depth] * count)
            node_depths_total.append(depth)

        c = Counter(node_depths)  # distribution pondérée
        c2 = Counter(node_depths_total)  # nombre de noeuds uniques par depth

        c_mean = {}
        for depth in c:
            if depth in c2 and c2[depth] > 0:
                c_mean[depth] = c[depth] / c2[depth]

        depths = sorted(c.keys())
        counts = [c[d] for d in depths]

        depths_mean = sorted(c_mean.keys())
        counts_mean = [c_mean[d] for d in depths_mean]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(depths, counts, width=0.6, color="darkorange", alpha=0.7, label="Raw count (c)")

        # Courbe moyenne normalisée
        # ax.plot(depths_mean, counts_mean, color="blue", marker="o", label="Normalized mean (c_mean)")

        ax.set_xticks(depths)
        ax.set_xlim(min(depths) - 1, max(depths) + 1)

        ax2 = ax.twinx()

        ax2.plot(depths_mean, counts_mean, color="blue", marker="o", label="Normalized mean (c_mean)")
        ax2.set_ylabel('Normalized mean (c_mean)', fontsize=12, color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

        # --- Merge legends ---
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2)
 
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        # Labels
        ax.set_xlabel('Depth', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Node Depth Distribution (Raw vs Normalized)', fontsize=14)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()

        plt.tight_layout()
        plt.savefig("depth_distribution_overlay.jpg", dpi=150)


    def npmi(self, x, y, eps=1e-12) -> float:
        """
        Compute Normalized Pointwise Mutual Information (NPMI).
        Returns a float value between [-1;1] where:
        -1 => no occurrences of this pair
        0 => no correlation between them
        1 => high correlation between them

        See:
            Bouma 2009 Normalized PMI

        Args:
            x, y : nodes
            eps : small value to avoid log(0)

        Returns:
            float : NPMI score in [-1, 1]
        """

        if x > y:
            x, y = y, x

        # Counts
        cx = self.node_count.get(x, 0)
        cy = self.node_count.get(y, 0)
        cxy = self.pair_count.get((x, y), 0)

        if cxy == 0:
            return -1.0  # convention: no cooccurrence => minimum association

        # Probabilities
        px = cx / self.N
        py = cy / self.N
        pxy = cxy / self.N

        # Avoid division by zero
        px = max(px, eps)
        py = max(py, eps)
        pxy = max(pxy, eps)

        # PMI
        pmi = math.log(pxy / (px * py))

        # NPMI (pmi / joint self_information https://en.wikipedia.org/wiki/Information_content)
        npmi_value = pmi / (-math.log(pxy))

        return npmi_value


    def mean_npmi(self,
                  nodes: List,
                  eps: float=1e-12) -> float:
        """
        Compute a mean NPMI of all nodes combination
        """
        if len(nodes) <= 1:
            return 1.0
        unique_nodes = set(nodes)
        all_npmi = []
        for i, j in combinations(sorted(unique_nodes), 2):
            all_npmi.append(self.npmi(i, j, eps))
        return sum(all_npmi) / len(all_npmi)


if __name__ == "__main__":
    """
    Save the figure of labels representations of the corpus.
    """
    uats = []
    reader = Reader()
    for doc in reader.read_corpus(ignore_kailas=False,
                                        corpus_folder=ADS_CORPUS_DIR):

        uat = doc.uats
        uats.append(uat)


    coherence = Coherence(uats)