"""
Compute coherence between output nodes.
"""
import math
from typing import List
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from src.utils.config import ADS_CORPUS_DIR
from src.utils.corpus_loader import Reader

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

        plt.style.use('_mpl-gallery')
        # make data:
        x = 1 + np.arange(len(self.node_count))
        y = sorted(self.node_count.values(), reverse=True)

        # plot
        fig, ax = plt.subplots()

        ax.bar(x, y, width=1, edgecolor="blue", linewidth=0.7)

        ax.set(xlim=(0, 2411), xticks=np.arange(1, 2411),
               ylim=(0, 4000), yticks=np.arange(1, 4000))


        y_max = max(y)
        y_ticks = np.arange(0, y_max + 500, 500)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        
        ax.set_xlabel('Node Index (ranked by frequency)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Node Frequencies', fontsize=14)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig("nodes_distrib.jpg", dpi=150)


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
        print(nodes)
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