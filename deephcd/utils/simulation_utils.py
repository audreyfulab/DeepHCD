from deephcd.utils.utilities import sort_labels
from deephcd.utils.plot_utils import plot_adj, plot_nodes
from deephcd.utils.train_utils import modularity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sbn
import networkx as nx
import numpy as np
import pandas as pd
import torch 
import torch.nn.functional as F
import pdb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def compute_graph_STATs(A_all, comm_assign, layers, sp, **kwargs):
    
    mod = []
    node_deg = []
    deg_within = []
    deg_between = []
    
    indices_top, indices_mid, labels_df, sorted_true_labels_top, sorted_true_labels_middle = sort_labels(comm_assign)
    A_sorted_by_top = A_all[-1]
    #compute statistics for bottom layer
    mod.append(compute_modularity(Adj = A_sorted_by_top, 
                                  sort_labels = sorted_true_labels_top))
    node_deg.append(compute_node_degree(A_sorted_by_top))
    deg_within.append(compute_node_degree_within(A=A_sorted_by_top, 
                                                 comm_labels = sorted_true_labels_top))
    deg_between.append(compute_node_degree_between(A=A_sorted_by_top, 
                                                   comm_labels = sorted_true_labels_top))
    
    #compute middle layer statistics 
    if layers > 2:
        #A_sorted_by_middle = resort_graph(A_all[-1], indices_mid)
        
        mod.append(compute_modularity(Adj = A_sorted_by_top, 
                                      sort_labels = sorted_true_labels_middle))
        
        node_deg.append(compute_node_degree(A_sorted_by_top))
        
        deg_within.append(compute_node_degree_within(A=A_sorted_by_top, 
                                                     comm_labels = sorted_true_labels_middle))
        
        deg_between.append(compute_node_degree_between(A=A_sorted_by_top, 
                                                       comm_labels = sorted_true_labels_middle))
        print('plotting middle graph')
        print(labels_df)
        print(pd.DataFrame(sorted_true_labels_middle, columns = ['labels middle']))
        plot_nodes(A_sorted_by_top, 
                   sorted_true_labels_middle, 
                   sp+'middle_graph',
                   **kwargs)
        plot_adj(A_sorted_by_top, 
                 sp+'middle_graph_adj')
        
    print('plotting top graphs')
    plot_nodes(A_sorted_by_top, 
               sorted_true_labels_top, 
               sp+'top_graph',
               **kwargs)
    plot_adj(A_sorted_by_top,  
             sp+'top_graph_adj')
    
    
    return mod, node_deg, deg_within, deg_between




# wrapper around draw_networx for diGraphs
def plot_diGraph(fig, ax, G, layer = 0, color_labels = None, 
                 draw_edge_weights = True, return_fig = False):
    
    pos = nx.circular_layout(G)
    if len(G.nodes) >= 10 and len(G.nodes) <= 100:
        ns = 500
    elif len(G.nodes) < 10:
        ns = 700
    else:
        ns = 300
    nx.draw_networkx(G, ax = ax, 
                     node_size = ns, 
                     pos = pos,
                     with_labels = True,
                     arrowsize = 40,
                     node_color = [int(str(i).split('_')[layer]) for i in G.nodes])
    if draw_edge_weights:
        nx.draw_networkx_edge_labels(G,
                                     pos,
                                     edge_labels=nx.get_edge_attributes(G,'weight'),
                                     font_size=10,
                                     ax = ax)
    
    if return_fig:
        return fig







#computes the modularity of an adjacency matrix based on a set of community 
#labels
def compute_modularity(Adj, sort_labels):
    A_tensor = torch.tensor(Adj).to(torch.float64)
    labels_tensor = torch.Tensor(sort_labels).to(torch.int64)
    P = F.one_hot(labels_tensor).to(torch.float64)
    mod = modularity(A_tensor, P)
    
    return mod.cpu().detach().numpy()






#generic function compute node degree
def compute_node_degree(A):
    deg = ((1/2)*A.sum())/A.shape[0]
    return deg
    





#this function computes the average node degree within communities given by
#a set of supplied community assignment labels
def compute_node_degree_within(A, comm_labels):
    N = A.shape[0]
    deg_list = []
    comms = np.unique(comm_labels)
    for i in range(len(np.unique(comm_labels))):
        ix = np.arange(N)[comm_labels == comms[i]]
        A_temp = A[ix,:]
        A_temp2 = A_temp[:, ix]
        deg_list.append(A_temp2.sum()/2)
        
    return np.mean(deg_list)
 


#this function computes the average node degree between communities given by
#a set of supplied community assignment labels  
def compute_node_degree_between(A, comm_labels):
    N = A.shape[0]
    deg_list = []
    comms = np.unique(comm_labels)
    for i in range(0, len(comms)):
        for j in range(0, len(comms)):
            
            if i != j:
                ix1 = np.arange(N)[comm_labels == comms[i]]
                ix2 = np.arange(N)[comm_labels == comms[j]]
                A_temp = A[ix1,:]
                A_temp2 = A_temp[:, ix2]
                deg_list.append(A_temp2.sum()/2)
            
    return np.mean(deg_list)
                





# this function uses the beth hessian to compute the number of detectable 
# communities by spectral means - method from 
# Schaub et al - Hierarchical community structure in networks
def compute_beth_hess_comms(A):
    N = A.shape[0]
    Deg = np.diag(np.matmul(A, np.ones((N, 1))).reshape(N))
    avg_degree = np.matmul(np.matmul(np.ones((N,1)).T, A), np.ones((N, 1)))/N
    eta = np.sqrt(avg_degree)
    Bethe_Hessian = (np.square(eta)-1)*np.diag(np.ones(N))+Deg - eta*A
    eigvals = np.linalg.eigh(Bethe_Hessian)[0]
    k = np.sum(eigvals<0)
    return k






def post_hoc_embedding(graph, embed_X, data, probabilities, labels, truth, 
                       is_torch = True, include_3d_plots = False, ns = 35, size = 10, 
                       fs=14, save = False, path = '', cm = 'plasma', **kwargs):
    layers = len(labels)
    if layers > 1:
        layer_nms = ['Middle Layer','Top Layer']
    else:
        layer_nms = 'Top Layer'
    #convert torch items     
    if is_torch:
        graph = graph.detach().numpy()
        X = data.detach().numpy()
        IX = embed_X.detach().numpy()
        probs = [i.detach().numpy() for i in probabilities]
    if any([isinstance(i, torch.Tensor) for i in labels]):
        labels = [i.detach().numpy() for i in labels]
    
        
    
        
    num_nodes = X.shape[0]
    #plot node traced_labels
    
    
    #plots of embeddings/correlations and probabilities
    fig2, (ax3, ax4) = plt.subplots(2, 2, figsize = (15, 10))
    #heatmap of embeddings correlations
    sbn.heatmap(np.corrcoef(X), ax = ax3[0])
    ax3[0].set_title('Correlations Data')
    #heatmap input data correaltions
    sbn.heatmap(np.corrcoef(IX), ax = ax3[1])
    ax3[1].set_title('Correlation matrix embeddings')
    fig_nx, ax_nx = plt.subplots(1,2,figsize=(12,10))
    
    for i in range(0, layers):
        print('plotting t-SNE and PCA for '+layer_nms[i])
        #nx graph with nodes colored by prediction
        # G = nx.from_numpy_array(graph)
        # templabs = np.arange(0, graph.shape[0])
        # clust_labels = {list(G.nodes)[k]: templabs.tolist()[k] for k in range(len(labels[i]))}
        # nx.draw_networkx(G, node_color = labels[i], 
        #                  labels = clust_labels,
        #                  font_size = fs,
        #                  node_size = ns,
        #                  cmap = cm, ax = ax_nx[i])
        # ax_nx[i].set_title(layer_nms[i]+' Clusters on Graph')
        
        #tsne
        TSNE_data=TSNE(n_components=3, 
                        learning_rate='auto',
                        init='random', 
                        perplexity=3).fit_transform(X)
        #pca
        PCs = PCA(n_components=3).fit_transform(X)
        #node labels
        nl = np.arange(TSNE_data.shape[0])
        #figs
        fig, (ax1, ax2) = plt.subplots(2,2, figsize = (12,10))
        #tsne plot
        ax1[0].scatter(TSNE_data[:,0], TSNE_data[:,1], s = size, c = labels[i], cmap = cm)
        ax1[0].set_xlabel('Dimension 1')
        ax1[0].set_ylabel('Dimension 2')
        ax1[0].set_title(layer_nms[i]+' t-SNE Data (Predicted)')
        #adding node labels
            
        #PCA plot
        ax1[1].scatter(PCs[:,0], PCs[:,1], s = size, c = labels[i], cmap = cm)
        ax1[1].set_xlabel('Dimension 1')
        ax1[1].set_ylabel('Dimension 2')
        ax1[1].set_title(layer_nms[i]+' PCA Data (Predicted)')
        #adding traced_labels
            
        
        if not isinstance(truth, type(None)):
            #TSNE and PCA plots using true cluster labels
            #tsne truth
            ax2[0].scatter(TSNE_data[:,0], TSNE_data[:,1], s = size, c = truth[i], cmap = cm)
            ax2[0].set_xlabel('Dimension 1')
            ax2[0].set_ylabel('Dimension 2')
            ax2[0].set_title(layer_nms[i]+' t-SNE Data (Truth)')
    
                
            #pca truth
            ax2[1].scatter(PCs[:,0], PCs[:,1], s = size, c = truth[i], cmap = cm)
            ax2[1].set_xlabel('Dimension 1')
            ax2[1].set_ylabel('Dimension 2')
            ax2[1].set_title(layer_nms[i]+' PCA Data (Truth)')
        # if num_nodes < 100:
        #     [ax1[0].text(i, j, f'{k}', fontsize=fs, ha='right') for (i, j, k) in zip(TSNE_data[:,0], TSNE_data[:,1], nl)]
        #     [ax1[1].text(i, j, f'{k}', fontsize=fs, ha='right') for (i, j, k) in zip(PCs[:,0], PCs[:,1], nl)]
        #     [ax2[0].text(i, j, f'{k}', fontsize=fs, ha='right') for (i, j, k) in zip(TSNE_data[:,0], TSNE_data[:,1], nl)]
        #     [ax2[1].text(i, j, f'{k}', fontsize=fs, ha='right') for (i, j, k) in zip(PCs[:,0], PCs[:,1], nl)]
        
        sbn.heatmap(probs[i], ax = ax4[i])
        ax4[i].set_title(layer_nms[i]+' Probabilities')
        
        
        fig3d_pcs = plt.figure(figsize=(12,10))
        ax3d = plt.axes(projection='3d')
        ax3d.scatter3D(PCs[:,0], PCs[:,1], PCs[:,2], 
                      c=labels[i], cmap='plasma')
        ax3d.set_title('PCA (predicted) '+layer_nms[i])
        ax3d.set_xlabel('Dimension 1')
        ax3d.set_ylabel('Dimension 2')
        ax3d.set_zlabel('Dimension 3')




        fig3d_tsne = plt.figure(figsize=(12,10))
        ax3d = plt.axes(projection='3d')
        ax3d.scatter3D(TSNE_data[:,0], TSNE_data[:,1], TSNE_data[:,2], 
                      c=labels[i], cmap='plasma')
        ax3d.set_title('TSNE (predicted) '+layer_nms[i])
        ax3d.set_xlabel('Dimension 1')
        ax3d.set_ylabel('Dimension 2')
        ax3d.set_zlabel('Dimension 3')
        
        #save plot by layer
        if save == True:
            fig.savefig(path+layer_nms[i]+'_tSNE_PCA_Plot.png', dpi = 300)
            fig3d_pcs.savefig(path+layer_nms[i]+'_3D_PCA_Plot.png', dpi = 300)
            fig3d_tsne.savefig(path+layer_nms[i]+'_3D_tSNE_Plot.png', dpi = 300)
            
    
    #save plots
    if save == True:
        fig2.savefig(path+'data_and_Probs.png', dpi = 300)







def plot_(X, cl = None, size = 10, cm = 'plasma'):
    
    TSNE_data=TSNE(n_components=3, 
                    learning_rate='auto',
                    init='random', 
                    perplexity=3).fit_transform(X)
    #pca
    PCs = PCA(n_components=3).fit_transform(X)
    #figs
    fig, (ax1, ax2) = plt.subplots(2,2, figsize = (12,10))
    #tsne plot
    ax1[0].scatter(TSNE_data[:,0], TSNE_data[:,1], s = size, c = cl, cmap = cm)
    ax1[0].set_xlabel('Dimension 1')
    ax1[0].set_ylabel('Dimension 2')
    ax1[0].set_title( 'TSNE')
    #adding node labels
        
    #PCA plot
    ax1[1].scatter(PCs[:,0], PCs[:,1], s = size, c = cl, cmap = cm)
    ax1[1].set_xlabel('Dimension 1')
    ax1[1].set_ylabel('Dimension 2')
    ax1[1].set_title('PCA')
    #adding traced_labels
    
    return fig, (ax1, ax2)
















