import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score,homogeneity_score, completeness_score, adjusted_rand_score, silhouette_score
from sklearn.neighbors import kneighbors_graph
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import scipy as spy
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import logging
from typing import Optional, Union, List, Literal, Dict, Any, Tuple


# custom logging config
def logging_config(logger_name: str = "Trainer", 
                   loglevel: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"] = "INFO",
                   log_to_file: bool = False,
                   file_path: str = "./training.log") -> logging.Logger:
    """
    Configures a logger with a given name and log level.

    Parameters:
    -----------
    logger_name : str, optional (default="Trainer")
        Name of the logger.
    loglevel : Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"], optional
        Logging level for the logger.
    log_to_file : bool, optional (default=False)
        Whether to save logs to a file.
    file_path : str, optional (default="./training.log")
        Path to save the log file if log_to_file=True.

    Returns:
    --------
    logger : logging.Logger
        Configured logger instance.

    Example:
    --------
    >>> logger = logging_config("MyTrainer", loglevel="DEBUG", log_to_file=True)
    >>> logger.info("Training started")
    """
    logger = logging.getLogger(logger_name)
    
    # Convert string loglevel to logging module level
    numeric_level = getattr(logging, loglevel.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Avoid adding duplicate handlers
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        if log_to_file:
            fh = logging.FileHandler(file_path)
            fh.setLevel(numeric_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    
    return logger



#----------------------------------------------------------------
def pickle_data(data, filename, filepath):
    with open(filepath+filename+'.pkl', 'wb') as f:
        pickle.dump(data, f)
        
        
        
#----------------------------------------------------------------
def open_pickled(filename):
    with open(filename, 'rb') as f:
        file = pickle.load(f)
        
    return file


# this function uses the beth hessian to compute the number of detectable 
# communities by spectral means - method from 
# Schaub et al - Hierarchical community structure in networks
def compute_beth_hess_comms(A: torch.Tensor):
    N = A.shape[0]
    Deg = torch.diag(torch.mm(A, torch.ones((N, 1))).reshape(N))
    avg_degree = torch.mm(torch.mm(torch.ones((N,1)).T, A), torch.ones((N, 1)))/N
    eta = torch.sqrt(avg_degree)
    Bethe_Hessian = (torch.square(eta)-1)*torch.diag(torch.ones(N))+Deg - eta*A
    eigvals = torch.linalg.eigh(Bethe_Hessian)[0]
    k = torch.sum(eigvals<0)
    return int(k)





#this function computes the number of communities k by chosen method
def compute_kappa(X: torch.Tensor, A: torch.Tensor, method: str = 'bethe_hessian', max_k: int = 50, save: bool = False, 
                  PATH: str = '/path/to/directory', verbose: bool = False):
    
    # bethe hessian spectral approach (fine partitions only)
    if method == 'bethe_hessian':
            kappa_middle = compute_beth_hess_comms(A)
            kappa_top = int(np.ceil(0.5*kappa_middle))
            
            if verbose:
                print(f'Beth Hessian estimated communities = {kappa_top}, {kappa_middle}')
            
    #elbow plot method
    elif method == 'elbow':
        inertias = []
        kappa_range = range(2, max_k+1)
        for i in kappa_range:
            result = KMeans(n_clusters=i, random_state=0).fit(X)
            inertias.append(result.inertia_)
        
        
        #compute change in inertia
        change_inertias = [float(np.abs(inertias[i] - inertias[i-1])) if i>0 else float(inertias[i]) for i in range(0, len(inertias))]
        
        #construct elbow plot
        fig, ax = plt.subplots(figsize = (12, 10))
        ax.plot(kappa_range, change_inertias)
        ax.set_xticks(kappa_range)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Change Inertia')
        ax.set_title('Elbow Plot')
        if save:
            fig.savefig(PATH+'Elbow_plot.pdf')
            
    #silouette optimization method
    elif method == 'silouette':
        scores = []
        kappa_range = range(2, max_k+1)
        for i in kappa_range:
            result = KMeans(n_clusters=i, random_state=0).fit(X)
            scores.append(silhouette_score(X, result.labels_))
        
        #determine peaks in silouette scores
        peaks, _ = find_peaks(scores, distance = 5)
        #eliminate peaks less than average silouette score
        score_idx = [(idx, scores[idx]) for idx in peaks]
        #first two peaks
        if len([score  for (idx, score) in score_idx if score > np.mean(scores)]) < 2:
            peak_scores = [(idx, score)  for (idx, score) in score_idx][:2]
        else:
            peak_scores = [(idx, score) for (idx, score) in score_idx if score > np.mean(scores)]
            
        #select kappa values
        kappa_top = kappa_range[peak_scores[0][0]]
        kappa_middle = kappa_range[peak_scores[1][0]]
        
        #construct silouette plot
        fig, ax = plt.subplots(figsize = (12, 10))
        ax.plot(kappa_range, scores)
        ax.set_xticks(kappa_range)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Silouette Score')
        ax.scatter([kappa_range[i[0]] for i in peak_scores], [i[1] for i in peak_scores], color = 'red')
        ax.axvline(x=kappa_top, label = 'best score', linestyle = 'dotted', linewidth = 2, color = 'red')
        ax.axhline(y=np.mean(scores), linestyle = '-.', linewidth = 2.5, color = 'black')
        ax.axvline(x=kappa_middle, linestyle = 'dotted', linewidth = 2, color = 'red')
        ax.set_title('Simulated Silouette Scores')
        #plt.show()
        if save:
            fig.savefig(PATH+'Silouette_scores.pdf')
        
        if verbose:
            print(f'Max Silouette scores: \n s = {peak_scores[0][1]:.4f} for k = {kappa_top} \n s = {peak_scores[1][1]:.4f} for k = {kappa_middle} ')
        
    else:
        raise ValueError(f'ERROR unrecognized value for argument method: "{method}" in compute_kappa()')
    
    return kappa_middle, kappa_top
    





#this simple function resorts a graph adjacency
#----------------------------------------------------------------
def resort_graph(A, sort_list):
    """
    A: adjacency matrix
    sort_list: a list of indices for sorting the adjacency A
    """
    A_temp1 = A[:,sort_list]
    A_sorted = A_temp1[sort_list, :]
    return A_sorted



#----------------------------------------------------------------
def easy_renumbering(labels):
    clusts = torch.unique(labels)
    new = torch.arange(len(clusts))
    for index, (i,j) in enumerate(zip(clusts, new)):
        labels[labels == int(i)] = int(j)
        
    return labels






#this function computes the homogeniety, completeness and NMI for a set of true and
#predicted cluster labels. 
#----------------------------------------------------------------
def node_clust_eval(true_labels, pred_labels, verbose = True):
    homogeneity, completeness, ari, nmi = (np.round(homogeneity_score(true_labels, pred_labels),4),
                                           np.round(completeness_score(true_labels, pred_labels),4),
                                           np.round(adjusted_rand_score(true_labels, pred_labels),4),
                                           np.round(normalized_mutual_info_score(true_labels, pred_labels),4))
    if verbose is True:
        print(f'\nHomogeneity = {homogeneity} \nCompleteness = {completeness} \nNMI = {nmi} \nARI = {ari}')
    
    return np.array([homogeneity, completeness, nmi, ari])
















#this function adjusts community level labels for all hierarchical layers
#to ensure the maximum label number is not greater the the number of allowable
#communities in heach layer
#----------------------------------------------------------------
def trace_comms(comm_list, comm_sizes):
    
    """
    comm_list: the list of community labels given as output from HCD model
    comm_sizes: the list of community sizes passed to HCD during model set up
    
    """
    
    comm_copy = comm_list.copy()
    comm_relabeled = comm_list.copy()
    layer =[]
    sizes = np.array(comm_sizes) - np.array([max(i) for i in comm_copy]) -1
    corrected_labs = [i+j for i,j in zip(comm_copy, sizes)]
    layer.append(corrected_labs[0])
    onehots = [F.one_hot(i) for i in corrected_labs]
    # for i in range(0, len(comm_copy)):
    #     #make sure the maximum community label is not greater than the number
    #     #communities for said layer. This is so that one_hot encoding doesn't
    #     #misinterpret the number of predicted communities
    #     comm_copy[i][comm_copy[i] == torch.max(comm_copy[i])] = comm_sizes[i]-1            
    #convert labels into one_hot matrix and trace assignments from layer back 
    #to original node size N 
    for i in range(1, len(comm_list)):
        layer.append(torch.linalg.multi_dot(onehots[:(i+1)]).argmax(1))
    
    comm_relabeled = layer.copy()
    for i in range(0, len(comm_list)):
        clusts = np.unique(layer[i])
        newlabs = np.arange(len(clusts))
        for j in range(0, len(clusts)):
            comm_relabeled[i][layer[i] == clusts[j]] = newlabs[j]
            
    return corrected_labs, comm_relabeled, layer
















# a simple function for quickly combining predicted and true community labels
# into a dataframe for plotting
#----------------------------------------------------------------
def gen_labels_df(pred_comms_list, truth, sorting, sort = True):
    
    """
    pred_comms_list: the predicted communities as output by trace_comms
    truth: the true labels (assumes they have sorted according to 'sorting')
    sorting: a list of indices for sorting the nodes in a particular order
    """
    pred_comms_list_cp = pred_comms_list.copy()
    num_layers = np.arange(0, len(pred_comms_list_cp))
    layer_nm = ['Communities'+'_Layer'+str(i) for i in num_layers]
    
    for i in range(0, len(pred_comms_list)):
        if sort == True:
            pred_comms_list_cp[i] = pred_comms_list_cp[i].cpu().detach().numpy()[sorting]
        else:
            pred_comms_list_cp[i] = pred_comms_list_cp[i].cpu().detach().numpy()
    
    for j in range(0, len(truth)):
        if len(truth[j]) > 0:    
            pred_comms_list_cp.append(truth[j])
            layer_nm.append('Truth_layer_'+str(j))
    df2 = pd.DataFrame(pred_comms_list_cp).T
    df2.columns = layer_nm
    
    return df2
















#utility to build the whole hierarchy adjacency from h1,h2,and h3 graphs
#----------------------------------------------------------------
def build_true_graph(file='filename'):
    
    """
    This function constructs the true graph of the hierarchical simulated data
    
    """
    
    obj = np.load(file)
    num_files = len(obj.files)
    num_layers = (num_files - 2)/2
    nodes_layer_i = []
    print('number of layers detected = {}, items in file = {}'.format(
        num_layers, obj.files
        ))
    for i in np.arange(num_layers)[::-1]:
        nodes_layer_i.append(len(obj['layer'+str(int(i+1))]))
    
    
    total_nodes = sum(nodes_layer_i)
    
    adj = np.zeros(shape = [total_nodes, total_nodes])
    
    for i, j in zip(np.arange(num_layers), np.arange(num_layers)[::-1]):
        if i == 0:
            adj[:nodes_layer_i[int(i)], :nodes_layer_i[int(i)]] = obj['adj_layer'+str(int(j+1))]
        else:
            splice_idx = [sum(nodes_layer_i[:int(i)]), sum(nodes_layer_i[:int(i+1)])] 
            print('layer = {}, Adj Splice = {}'.format(
                str(int(j+1)), splice_idx
                ))
            adj[splice_idx[0]:splice_idx[1],splice_idx[0]:splice_idx[1]] = obj['adj_layer'+str(int(j+1))]
            
    return adj






#----------------------------------------------------------------
def sort_labels(labels):
    #pdb.set_trace()
    true_labels = pd.DataFrame(labels, columns = ['Nodes'])
    new_list = []
    l = len(true_labels['Nodes'])
    for i in np.arange(l):
        new_list.append(list(map(int, true_labels['Nodes'][i].split('_'))))
        
    new_list_array = np.array(new_list)
    new_true_labels = pd.DataFrame(new_list_array[:,:2])
    new_true_labels.columns = ['toplabs', 'nodelabs']
    clusts = np.unique(new_true_labels['toplabs'])
        
    indices_for_clusts = []
    #extract indices for nodes belonging to each cluster
    for i in clusts:
        indices_for_clusts.append(new_true_labels[new_true_labels['toplabs'] == i].index.tolist())
        

    #pe = np.load(path+'gexp.npy').transpose()
    #reorganize pe so that nodes are sorted according to clusters 0,1,2..etc
    flat_list_indices = []
    for i in clusts:
        flat_list_indices.extend(indices_for_clusts[i])
        
        
    #construct labels for middle layer nodes
    if new_list_array.shape[1]>2:
        #num_middle_nodes = new_list_array.shape[0]/len(np.unique(new_list_array[:,1]))
        #temp = [np.repeat(i,len(np.unique(new_list_array[:,1]))).tolist() for i in np.arange(num_middle_nodes)]
        #middle_labels = np.array([int(i[0]) for i in np.array(temp).reshape((l,1)).tolist()])
        temp = [int(str(i[0])+str(i[1])) for i in new_list]
        middle_labels = np.array(temp.copy())
        midclusts = np.unique(temp)
        newclusts = np.arange(len(midclusts))
        for i in range(0, len(midclusts)):
            middle_labels[middle_labels == midclusts[i]] = newclusts[i]
            
        new_true_labels['middlelabs'] = middle_labels
        indices_for_clusts2 = []
        flat_list_indices2 = []
        #extract indices for nodes belonging to each cluster
        for i in newclusts:
            indices_for_clusts2.append(new_true_labels[new_true_labels['middlelabs'] == i].index.tolist())
            flat_list_indices2.extend(indices_for_clusts2[i])
        
        sorted_true_labels_middle = middle_labels[flat_list_indices2]
    else:
        flat_list_indices2 = []
        sorted_true_labels_middle = []


    #the true labels sorted by cluster
    sorted_true_labels_top = np.array(new_true_labels['toplabs'][flat_list_indices])

    return flat_list_indices, flat_list_indices2, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle







#----------------------------------------------------------------
def LoadData(filename):
    unparsed_labels = pd.read_csv(filename+'_gexp.csv', index_col=0).columns.tolist()
    flat_list_indices, flat_list_indices2, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle = sort_labels(unparsed_labels)
    pe = np.load(filename+'_gexp.npy').transpose()
    true_adj = build_true_graph(filename+'.npz')
    G = nx.from_numpy_array(true_adj)
    G = G.to_undirected()
    true_adj_undi = nx.adjacency_matrix(G).toarray()
    #the true labels sorted by cluster

    return pe, true_adj_undi, flat_list_indices, flat_list_indices2, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle
    




#----------------------------------------------------------------
def corr_dist(u, v, w=None, centered=True):
    """
    Compute the correlation distance between two 1-D arrays.
    The correlation distance between `u` and `v`, is
    defined as
    .. math::
        1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                  {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
    where :math:`\\bar{u}` is the mean of the elements of `u`
    and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0
    centered : bool, optional
        If True, `u` and `v` will be centered. Default is True.
    Returns
    -------
    correlation : double
        The correlation distance between 1-D array `u` and `v`.
    """
    #u = _validate_vector(u)
    #v = _validate_vector(v)
    #if w is not None:
    #    w = _validate_weights(w)
    if centered:
        umu = np.average(u, weights=w)
        vmu = np.average(v, weights=w)
        u = u - umu
        v = v - vmu
    uv = np.average(u * v, weights=w)
    uu = np.average(np.square(u), weights=w)
    vv = np.average(np.square(v), weights=w)
    dist = 1.0 - np.square(uv / np.sqrt(uu * vv))
    # Return absolute value to avoid small negative value due to rounding
    return np.abs(dist) 




#----------------------------------------------------------------
def get_pcor(data, cutoff = 0.05):
    
    """
        this function takes in a numpy array with features in the rows and observations
        in the columns and computes the precision and partial correlation matrices. It
        then preforms hypothesis testing on the partial correaltions to return the undirected 
        adjacency matrix
    """
    
    #get shapes
    n = data.shape[1]
    #calc precision matrix
    H = np.linalg.inv(np.cov(data))
    hdim = H.shape[0]
    S = hdim-2
    #preallocate
    pcor_mat = np.array([1.0]*np.square(hdim)).reshape((hdim, hdim))
    stat_mat = np.array([0.0]*np.square(hdim)).reshape((hdim, hdim))
    pval_mat = np.array([0.0]*np.square(hdim)).reshape((hdim, hdim))
    adj = np.array([0]*np.square(hdim)).reshape((hdim, hdim))
    # calculate partial correlation and stat, and pvalue
    print('Calculating partial correaltion matrix...')
    for i in np.arange(hdim):
        for j in np.arange(hdim):
            if i != j:
                pcor = -((H[i,j])/(np.sqrt(H[i,i])*np.sqrt(H[j,j])))
                pcor_mat[i,j] = pcor 
                stat = (np.sqrt(n - S - 3)/2)*np.log((1+pcor_mat[i,j])/(1-pcor_mat[i,j]))
                stat_mat[i,j] = stat
                pval = 2*(1-spy.stats.norm.cdf(np.absolute(stat_mat[i,j])))
                pval_mat[i,j] = pval
                
                
            
                if pval_mat[i,j]<cutoff:
                    adj[i,j] = 1
            
    return pcor_mat, stat_mat, pval_mat, adj






#----------------------------------------------------------------
def get_input_graph(X = None, method = ['KNN','Presicion','Correlation', 'DAG-GNN'], 
                    K = 30, metric = '1-R^2', alpha = 0.05, r_cutoff = 0.2):
    
    
    '''
    This function computes the input graph to the GNN using the desired method
    
    '''
    
    
    
    if method == 'KNN':
        if metric == '1-R^2':
            A = kneighbors_graph(X, n_neighbors = K, metric = corr_dist)
        else:    
            #calculate the kneighbors graph
            A = kneighbors_graph(X, n_neighbors = K, metric = metric)
        
        #convert to adjacency (this matrix is directed)
        A_adj = A.toarray()
        
    if method == 'Precision':
        
        pcors, stats, pvals, A_adj = get_pcor(X, cutoff=alpha)
        
    if method == 'Correlation':
        #get the absolute correlations
        cormat = np.absolute(np.corrcoef(X))
        A_adj = np.copy(cormat)
        A_adj[A_adj>r_cutoff] = 1
        A_adj[A_adj<=r_cutoff] = 0
        np.fill_diagonal(A_adj, 0)
        
        
    #if method == 'DAG-GNN':
        
        
        
    #also get graph
    A_graph = convert_adj_to_graph(adj=A_adj)
    #get the undirected knn
    A_undi = nx.to_numpy_array(A_graph)
    return A_graph, A_undi




#----------------------------------------------------------------
def convert_adj_to_graph(adj):
    graph = nx.from_numpy_array(adj, parallel_edges=False, create_using=nx.Graph)
    #set as undirected graph
    undi_h3 = graph.to_undirected()
    #extract original node labels
    gx = nx.Graph().to_undirected()
    #add in the nodes from undi_h3
    gx.add_nodes_from(undi_h3.nodes)
    #a small function to bind two lists into a list of tuples
    
    #extract the two componets of the edgelist from undi_h3 into separate lists
    list1 = list(nx.to_pandas_edgelist(undi_h3)['source'])
    list2 = list(nx.to_pandas_edgelist(undi_h3)['target'])
    #merge the lists into a list of 2-tuples i.e (source, target)
    edge_tuples = merge(list1, list2)
    #add the edges into the graph
    gx.add_edges_from(edge_tuples)
    return gx









#---------------------------------------------------------------- 
def merge(list1, list2):
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list






def get_layered_performance(k, S_relab, true_labels):
    perf_layers = []
    for i in range(0, k):
        eval_metrics = node_clust_eval(true_labels=true_labels[i],
                                       pred_labels=S_relab[i].cpu(), 
                                       verbose=False)
        perf_layers.append(eval_metrics.tolist())
    
    return perf_layers













