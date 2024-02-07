"""
We construct a graph that represents the dependency of the economic indicators on each other. 
One way to do this is to use the distance or correlation matrix of the indicators and apply a threshold to determine the edges. 
We may also need to assign weights to the edges based on some measure of similarity or influence.
"""
import typing
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes

def compute_adjacency_matrix(
    economic_indicators: np.ndarray,
    sigma2: float,
    epsilon: float,
):
    """Computes the adjacency matrix from economic indicators matrix.
    It uses the formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing
    to compute an adjacency matrix from the economic indicators matrix.
    The implementation follows that paper.

    Args:
        economic_indicators: np.ndarray of shape `(num_time_steps, num_indicators)`.
            Entry `i,j` of this array is the value of indicator `j` at time `i`.
        sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.
        epsilon: A threshold specifying if there is an edge between two nodes.
            Specifically, `A[i,j]=1` if `np.exp(-w2[i,j] / sigma2) >= epsilon`
            and `A[i,j]=0` otherwise, where `A` is the adjacency matrix and
            `w2=economic_indicators * economic_indicators`

    Returns:
        A boolean graph adjacency matrix.
    """
    num_indicators = economic_indicators.shape[1]
    # Compute the pairwise correlation coefficients between the indicators
    correlation_matrix = np.corrcoef(economic_indicators, rowvar=False)
    # Compute the square distances matrix from the correlation matrix
    square_distances = np.square(correlation_matrix)
    # Apply the Gaussian input_sequence_lengthkernel to the square distances matrix
    w2 = square_distances / sigma2
    return (np.exp(-w2) >= epsilon) * (1 - np.identity(num_indicators))

def plot_graph(graph: GraphInfo, correlation_matrix: np.ndarray):
    """Plots the graph with correlation coefficients as edge labels."""
    plt.figure(figsize=(8, 8))
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(zip(*graph.edges))
    
    # Filter out isolated nodes
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    
    pos = nx.spring_layout(G, k=.5)  # Adjust the k parameter for node spacing
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=200)
    nx.draw_networkx_edges(G, pos, width=1, alpha=.7)
    nx.draw_networkx_labels(
        G, pos, font_size=10, font_color="black", font_weight="bold"
    )
    edge_labels = {
        (u, v): f"{correlation_matrix[u, v]:.2f}" for u, v in G.edges
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=8
    )
    plt.axis("off")
    plt.show()
    