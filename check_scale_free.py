import pickle as pkl
import networkx as nx
import powerlaw
import matplotlib.pyplot as plt

def call(dataset):
    graph = pkl.load(open("ARGA/arga/data/ind.{}.graph".format(dataset), 'rb'), encoding='latin1')
    
    G = nx.from_dict_of_lists(graph)
    # adj = nx.adjacency_matrix(G)

    fig, axs = plt.subplots(2)
    fig.suptitle(dataset)

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    axs[0].hist(degree_sequence)
    
    fit = powerlaw.Fit(degree_sequence, xmin=1)
    fit.plot_pdf(color='b', linewidth=2, ax=axs[1])
    fit.power_law.plot_pdf(color='g', linestyle='--', ax=axs[1])

    plt.show()

# call('cora')
# call('citeseer')
call('pubmed')