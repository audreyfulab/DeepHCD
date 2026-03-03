import argparse
import sys
import os
import tracemalloc
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from typing import Literal
from deephcd.model.model import HCD
from deephcd.model.train import Trainer
from deephcd.utils.train_utils import split_dataset
from deephcd.utils.utilities import compute_kappa,LoadData,get_input_graph

#expose paths to necessary files
#base_dir = "/Users/jordandavis/Documents/HGRN_repo_fork"

# Add necessary folders to the Python path
#sys.path.append(os.path.join(base_dir, "HGRN_software"))
#sys.path.append(os.path.join(base_dir, "Simulated Hierarchies"))
#sys.path.append(os.path.join(base_dir, "Simulated Hierarchies", "run_simulation_utils"))
#sys.path.append(os.path.join(base_dir, "model"))

def set_up_model_for_simulation_inplace(args, simargs, load_from_existing = False):
    
    if not load_from_existing:
        if not os.path.exists(simargs.savepath):
            os.makedirs(simargs.savepath)
            
        
        
        info_table = pd.DataFrame(columns = ['subgraph_type', 'connection_type', 'connection_prob','layers','StDev',
                                            'nodes_per_layer', 'edges_per_layer', 'subgraph_prob',
                                            'sample_size','modularity_top','avg_node_degree_top',
                                            'avg_connect_within_top','avg_connect_between_top',
                                            'modularity_middle','avg_node_degree_middle',
                                            'avg_connect_within_middle','avg_connect_between_middle'
                                            ])
        
        print(f'saving hierarchy to {simargs.savepath}')
        pe, gexp, nodes, edges, nx_all, adj_all, path, nodelabs, ori = simulate_graph(simargs)
        print('done.')
        print('computing statistics....')
        compute_stat_time_start = time.time()
        mod, node_deg, deg_within, deg_between = compute_graph_STATs(A_all = adj_all, 
                                                                    comm_assign = nodelabs, 
                                                                    layers = simargs.layers,
                                                                    sp = simargs.savepath,
                                                                    node_size = 60,
                                                                    font_size = 9,
                                                                    add_labels=True)
        compute_stat_time_end = time.time()
        print(f'Computation of Stats Time: ', compute_stat_time_end-compute_stat_time_start)
        print('*'*25+'top layer stats'+'*'*25)
        print('modularity = {:.4f}, mean node degree = {:.4f}'.format(
            mod[0], node_deg[0]
            )) 
        print('mean within community degree = {:.4f}, mean edges between communities = {:.4f}'.format(
            deg_within[0], deg_between[0] 
            ))
        if simargs.layers > 2:
            print('*'*25+'middle layer stats'+'*'*25)
            print('modularity = {:.4f}, mean node degree = {:.4f}'.format(
                mod[1], node_deg[1]
            )) 
            print('mean within community degree = {}, mean edges between communities = {}'.format(
                deg_within[1], deg_between[1] 
                ))
            print('*'*60)
            
            
        cp_dict = {'middle': simargs.connect_prob_middle, 'bottom': simargs.connect_prob_bottom}
        
        if simargs.layers == 3:
            
            row_info = [simargs.subgraph_type, simargs.connect, cp_dict, simargs.layers, simargs.SD,
                        tuple(nodes),tuple(edges),simargs.subgraph_prob, simargs.sample_size,
                        mod[0], node_deg[0], deg_within[0], deg_between[0], 
                        mod[1], node_deg[1], deg_within[1], deg_between[1]]
        else:
            row_info = [simargs.subgraph_type, simargs.connect, cp_dict, simargs.layers, simargs.SD,
                        tuple(nodes),tuple(edges), simargs.sample_size,
                        mod[0], node_deg[0], deg_within[0], deg_between[0], 
                        'NA', 'NA', 'NA', 'NA']
            
        print(pd.DataFrame(row_info))
        
        print('done')
        print('saving hierarchy statistics...')
        info_table.loc[0] = row_info
        info_table.to_csv(simargs.savepath+'intermediate_examples_network_statistics.csv')
        np.savez(simargs.savepath+'intermediate_examples_network_statistics.npz', data = info_table.to_numpy())
        print('done')
        
        print('Simulation Complete')       
        print(f'Loading simulated data from directory: {simargs.savepath}')
    pe, true_adj_undi, indices_top, indices_middle, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle = LoadData(filename=simargs.savepath)
    
    #combine target labels into list
    print('Read in expression data of dimension = {}'.format(pe.shape))
    if simargs.layers == 2:
        target_labels = [sorted_true_labels_top, []]
        #sort nodes in expression table 
        pe_sorted = pe[indices_top,:]
    else:
        target_labels = [sorted_true_labels_top, 
                            sorted_true_labels_middle]
        #sort nodes in expression table 
        pe_sorted = pe[indices_middle,:]
     
    #nodes and attributes
    nodes, attrib = pe.shape
    X = torch.Tensor(pe_sorted)
    #generate input graphs
    if args.use_true_graph:
        A = (torch.Tensor(true_adj_undi[:nodes,:nodes])+torch.eye(nodes))
    else:    
        in_graph, in_adj = get_input_graph(X = pe_sorted, 
                                           method = 'Correlation', 
                                           r_cutoff = args.correlation_cutoff)

        A = torch.Tensor(in_adj)+torch.eye(nodes)
            
    return X, A, target_labels
def main():
    parser = argparse.ArgumentParser(description='Model Parameters')
    parser.add_argument('--dataset', type=str, choices=['complex', 'intermediate', 'toy', 'cora', 'pubmed', 'regulon.EM', 'regulon.DM', 'generated'], required=False, default='generated', help='Dataset selection. When dataset is "generated" the data are simulated accord to simulation argments')
    parser.add_argument('--parent_distribution', type=str, choices=['same_for_all', 'unequal'], required=False, help='Parent distribution. when "same_for_all" all parent nodes are simulated from a N(0,1) distribution')
    parser.add_argument('--read_from', type=str, default='local', choices = ['local','cluster'], help='Path to read data from. if "local" data is read from the "path/to/repo/Simulated Hierarchies/DATA/')
    parser.add_argument('--which_net', type=int, default=0, help='Network selection. Used for pre-generated datasets only')
    parser.add_argument('--use_true_graph', type=bool, default=True, help='Sets the input graph to be the true graph')
    parser.add_argument('--correlation_cutoff', type=float, default=0.2, help='The minimum correlation required for an edge to be added between two nodes')
    parser.add_argument('--use_method', type=str, default='top_down', choices=['top_down','bottom_up'], help='method for uncovering the hierarchy. "top_down" = divisive clustering while "bottom_up" = additive clustering' )
    parser.add_argument('--use_softKMeans_top', type=bool, default=False, help='If true, the top layer is inferred with a softKMeans layer')
    parser.add_argument('--use_softKMeans_middle', type=bool, default=False, help='If true, the middle layer is inferred with a softKMeans layer (currently broken)')
    parser.add_argument('--gamma', type=float, default=1, help='Attribute reconstruction loss hyperparameter')
    parser.add_argument('--delta', type=float, default=1, help='Modularity loss hyperparameter')
    parser.add_argument('--lambda_', nargs='+', type=Literal[float], default=[1.0, 1.0], help='Clustering loss hyperparameters. First value is for middle layer, second value is for top layer')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default = 0.2, help='Dropout rate')
    parser.add_argument('--use_batch_learning', type=bool, default=False, help='If true data is divided into batches for training according to batch_size')
    parser.add_argument('--batch_size', type = int, default=64, help='size of batches used when use_batch_learning == True')
    parser.add_argument('--training_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--steps_between_updates', type=int, default=10, help='Number of training epochs before each update')
    parser.add_argument('--resolution', nargs='+', type=Literal[float], default=None, help='The resolution regularization parameter in the modularity loss calculation')
    parser.add_argument('--AE_hidden_size', nargs='+', type=Literal[int], default=[256, 128, 64], help='Hidden layer sizes for GATE')
    parser.add_argument('--LL_hidden_size', nargs='+', type=Literal[int], default = [64, 64], help='Sizes for additional learning layers applied to GATE embedding before inference. Only used when "')
    parser.add_argument('--AE_operator', type=str, choices=['GATConv', 'GATv2Conv', 'SAGEConv'], default='GATv2Conv', help='The type of layer that should be used in the graph autoencoder architecture')
    parser.add_argument('--COMM_operator', type=str, choices=['None', 'Conv2d', 'Linear', 'GATConv', 'GATv2Conv', 'SAGEConv'], default='Linear', help='The type of layer that should be used in the community detection module')
    parser.add_argument('--use_true_communities', type=bool, default=True, help='Use true communities')
    parser.add_argument('--community_sizes', nargs='+', type=Literal[int], default=[15, 5], help='Specifies the (max) number of communities to be inferred in the middle and top layers respectively')
    parser.add_argument('--activation', type=str, choices=['LeakyReLU', 'Sigmoid'], required=False, help='Activation function (ignore)')
    parser.add_argument('--use_gpu', type=bool, default=True, help='When True, HCD defaults to device="gpu" if torch.cuda.is_available() is True else device is set to "cpu" ')
    parser.add_argument('--verbose', type=bool, default=True, help='when True, additional plots are output in training updates')
    parser.add_argument('--return_result', type=str, choices=['best_perf_top', 'best_perf_mid', 'best_loss'], required=False, default=True, help='When True, all model training results are returned')
    parser.add_argument('--save_results', type=bool, default=False, help='When True, results are saved to path at "sp" ')
    parser.add_argument('--set_seed', type=bool, default=True, help='Sets a random seed for training the model')
    parser.add_argument('--seed', dest='seed', type=int, default=555, help='Seed number for training model')
    parser.add_argument('--sp', type=str, default='/my/save/path/', help='Specified path to save directory')
    parser.add_argument('--plotting_node_size', type=int, default=25, help='Node size - used in plotting')
    parser.add_argument('--fs', type=int, default=10, help='Font size - used in plotting')
    parser.add_argument('--run_louvain', type=bool, default=True, help='When True, runs Louvain method on input graph for comparison')
    parser.add_argument('--run_kmeans', type=bool, default=True, help='When True, runs KMeans on gene expression for comparison')
    parser.add_argument('--run_hc', type=bool, default=True, help='When True, runs hierarchical clustering on gene expression for comparison')
    parser.add_argument('--use_multihead_attn', type=bool, default=True, help='When True, Multihead attention is applied to all GATE graph attention layers')
    parser.add_argument('--attn_heads', type = int, default = 5, help='The number of attention heads for multihead attention')
    parser.add_argument('--normalize_layers', type=bool, default = True, help='When True, the output of each neural layer is normalized before activation')
    parser.add_argument('--normalize_input', type=bool, default=True, help='When True, the input features are normalized before model fitting')
    parser.add_argument('--split_data', type=bool, default=False, help='When True, data is split into training and validation sets')
    parser.add_argument('--early_stopping', type=bool, default=True, help='If True, early stopping is used during training')
    parser.add_argument('--patience', type=int, default=5, help='Number of consecutive epochs with no improvement after which training will stop (only relevant if early stopping is enabled).')
    parser.add_argument('--train_val_size', nargs='+', type=Literal[float], default = [0.8, 0.2], help='Specifies the fraction of data in training and validation sets respectively')
    parser.add_argument('--post_hoc_plots', type=bool, default=True, help='When True, additional plots of results are made')
    parser.add_argument('--add_output_layers', type=bool, default=False, help ='When True, extra neural layers are added between the embedding and prediction layers')
    parser.add_argument('--make_directories', type=bool, default=False, help='If true directories are created using os.makedir()')
    parser.add_argument('--save_model', type=bool, default=False, help='When True, model is serialized and saved at args.sp as model.pth file')
    parser.add_argument('--compute_optimal_clusters', type=bool, default=True, help='When True, values of "community_sizes" is estimated using the method specified in "kappa_method"')
    parser.add_argument('--kappa_method', type=str, default='bethe_hessian', choices = ['bethe_hessian', 'elbow', 'silouette'], help='Method for determining the optimal number of communities for the top and middle layers of the hierarchy (currently "elbow" method does not work)')
    parser.add_argument('--load_from_existing', type=bool, default=False, help='When --dataset = "generated" and argument is True, data is read from same directory where previous graph was saved')
    args = parser.parse_args()



    # Simulation arguments
    parser2 = argparse.ArgumentParser(description='Simulation Parameters')
    parser2.add_argument('--connect', dest='connect', choices=['disc', 'full'], default='disc', type=str, help='Sets the top layer of the simulated hierarchy to be either all connected or all disconnected')
    parser2.add_argument('--connect_prob_middle', dest='connect_prob_middle', default=0.1, type=float, help='Sets the probability for edge creation between two nodes in the middle layer of the hierarchy')
    parser2.add_argument('--connect_prob_bottom', dest='connect_prob_bottom', default=0.01, type=float, help='Sets the probability for edge creation between two nodes in the bottom layer of the hierarchy')
    parser2.add_argument('--top_layer_nodes', dest='top_layer_nodes', default=5, type=int, help='Sets the number of nodes (i.e communities) in the top layer of the hierarchy')
    parser2.add_argument('--subgraph_type', dest='subgraph_type', default='small world', type=str, help='Sets the type of subgraph used to generate communities in the middle and bottom layers of the hierarchy')
    parser2.add_argument('--subgraph_prob', nargs='+', dest='subgraph_prob', default=(0.5, 0.2), type=tuple[float], help='Sets the probability of edge creating within small world subgraphs')
    parser2.add_argument('--nodes_per_super2', nargs='+', dest='nodes_per_super2', default=(3,3), type=tuple[float], help='sets the number of offspring for each node in the top layer of the hierarchy')
    parser2.add_argument('--nodes_per_super3', nargs='+', dest='nodes_per_super3', default=(20,20), type=tuple[float], help='Sets the number of offspring for each node in the bottom layer of the hierarchy')
    parser2.add_argument('--node_degree_middle', dest='node_degree_middle', default=3, type=int, help='sets the node degree for random graph subgraphs in the middle layer')
    parser2.add_argument('--node_degree_bottom', dest='node_degree_bottom', default=5, type=int, help='sets the node degree for random graph subgraphs in the middle layer')
    parser2.add_argument('--sample_size',dest='sample_size', default = 500, type=int, help='Sample size (people/cells) for simulated gene expression')
    parser2.add_argument('--layers',dest='layers', default = 3, type=int, help='Sets the number of layers in the hierarchy. Note that HCD currently only supports two or three layer hierarchies')
    parser2.add_argument('--SD',dest='SD', default = 0.1, type=float, help='Sets the standard deviation for all simulated genes')
    parser2.add_argument('--common_dist', dest='common_dist',default = False, type=bool, help='When True, all parent nodes of the network are simulated from a N(0,sigma) distribution. When False, all parent nodes are simulated from N(mu_k, sigma)')
    parser2.add_argument('--use_weighted_graph', dest='use_weighted_graph',default = False, type=bool, help='When True, a weigted network is generated according --within_edgeweights and --between_edgeweights')
    parser2.add_argument('--within_edgeweights', dest='within_edgeweights', nargs='+', default = (0.5, 0.8), type=tuple[float], help='Tuple giving the minimum and maximum weight for an edge between nodes in the same community')
    parser2.add_argument('--between_edgeweights', dest='between_edgeweights', nargs='+', default = (0, 0.2), type=tuple[float], help='Tuple giving the minimum and maximum weight for an edge between nodes in different communities')
    parser2.add_argument('--set_seed', dest='set_seed', default=False, type=bool, help='When True, a random seed is set for generating the network')
    parser2.add_argument('--seed_number', dest='seed_number',default = 555, type=int, help='random seed for network generation')
    parser2.add_argument('--force_connect', dest='force_connect', default=True, type=bool, help='Enforces connectivity between communities: If True, ensures at least one edge exists between nodes from communities that were connected in the previous layer')
    parser2.add_argument('--savepath', dest='savepath', default='./', type=str, help='PATH name specifying where the simulated network should be saved')
    parser2.add_argument('--mixed_graph', dest='mixed_graph', default=False, type=str, help='When True, a graph with mixed small world, scale free, and random graph topolgies is generated')
    sim_args = parser2.parse_args()

    print('successful args add')
    sim_args.subgraph_type = 'small world'
    print(sim_args)

    sim_args.set_seed = True
    sim_args.seed_number = 555
    #sim_args.savepath = '/Users/jordandavis/Desktop/HGRN_repo/attempt_on_test7/test6/'
    sim_args.savepath = '/Users/audreyq.fu/Documents/Programs/DeepHCD/examples/very_small_graph_150/'

    #output save settings
    args.sp = '/Users/audreyq.fu/Documents/Programs/DeepHCD/examples/output/'
    args.use_gpu = True
    sim_args.use_multihead_attn = True
    args.save_results = True
    args.make_directories = True #this will automatically create the directories at args.sp and sim_args.savepath
    args.set_seed = 555 #sets a seed for training the model
    args.load_from_existing = True #this will ensure a new graph is simulated. When set to True, data retreiver looks for graph elements at directory sim_args.savepath
    args.dataset = 'generated' #sets the dataset to be simulated according arguments passed in sim_args



    #graph settings
    def scale_connection_prob(num_nodes, avg_degree_range=(4, 12)):
        low_c, high_c = avg_degree_range 
        low_p = low_c / num_nodes
        high_p = high_c / num_nodes
        return [np.random.uniform(low_p, high_p) for _ in range(2)]

    sim_args.connect = 'disc'
    sim_args.nodes_per_super2 = (1,1)
    sim_args.nodes_per_super3 = (4,9)
    n1 = sim_args.top_layer_nodes
    k2 = sim_args.nodes_per_super2[0]
    k3 = sim_args.nodes_per_super3[0]

    n2 = n1 * k2  # Middle layer
    n3 = n2 * k3  # Bottom layer

    sim_args.node_degree_middle = 3
    sim_args.node_degree_bottom = 15

    sim_args.connect_prob_middle = scale_connection_prob(n2, avg_degree_range=(.2, .6))
    sim_args.connect_prob_bottom = scale_connection_prob(n3, avg_degree_range=(.3, .10))


    #model set up
    args.early_stopping = True
    args.patience = 2
    args.use_method = "top_down"
    args.use_batch_learning = True
    args.batch_size = 64
    args.use_softKMeans_top = False
    args.use_softKMeans_middle = False
    args.add_output_layers = False
    args.AE_operator = 'GATv2Conv'
    args.COMM_operator = 'None'
    args.attn_heads = 5
    args.dropout_rate = 0.2
    args.normalize_input = True
    args.normalize_layers = True
    args.AE_hidden_size = [256, 128] 
    args.gamma = 2
    args.delta = 10
    args.lambda_ = [1/60, 1/20]
    args.learning_rate = 1e-3
    args.community_sizes = [15, 5]
    args.compute_optimal_clusters = True #this overrides args.community_sizes and estimates k_middle, k_top according to "kappa_method"
    args.kappa_method = 'bethe_hessian'

    #training set up
    args.training_epochs = 50
    args.steps_between_updates = 10
    args.use_true_graph = False 
    args.correlation_cutoff = 0.2
    args.return_result = 'best_loss'
    args.verbose = False
    args.run_louvain = True
    args.run_hc = True
    args.split_data = True
    args.train_val_size = [0.8, 0.2]

    tracemalloc.start()
    #generate data and train model - returns output object of class HCD_output
    '''
    >>> from model.model import HCD
    >>> from model.train import fit
    >>> model = HCD()  # Initialize your model
    >>> X = torch.rand(100, 20)  # Example feature matrix
    >>> A = torch.randint(0, 2, (100, 100))  # Example adjacency matrix
    >>> trainer = Trainer(model, X, A, epochs=50, lr=1e-3, batch_size=32, early_stopping=True, patience=5)
    '''
    device = 'cuda:'+str(0) if args.use_gpu and torch.cuda.is_available() else 'cpu'
    X, A, target_labels = set_up_model_for_simulation_inplace(args, sim_args, load_from_existing=args.load_from_existing)
    train, val_set = split_dataset(X, A, target_labels, args.train_val_size)
    X, A, target_labels = train
    nodes, attrib = X.shape
    if args.compute_optimal_clusters:
        comm_sizes = compute_kappa(X, A, method = args.kappa_method, save = args.save_results, PATH = args.sp, verbose = True)

    model = HCD(
                nodes, attrib, 
                method=args.use_method,
                use_kmeans_top=args.use_softKMeans_top,
                use_kmeans_middle=args.use_softKMeans_middle,
                ae_hidden_dims=args.AE_hidden_size,
                ll_hidden_dims = args.LL_hidden_size,
                comm_sizes=comm_sizes, 
                normalize_input = args.normalize_input,
                normalize_outputs = args.normalize_layers,
                ae_operator = args.AE_operator,
                comm_operator = args.COMM_operator,
                ae_attn_heads = args.attn_heads, 
                dropout = args.dropout_rate,
                use_output_layers = args.add_output_layers,
                heads=1
                ).to(device)
    
    trainer = Trainer(model, X, A, epochs=50, learning_rate=1e-3, batch_size=32, early_stopping=True, patience=5, validation_data=val_set, output_path=args.sp)
    trainer.fit(device)



if __name__ == "__main__":
    main()
#fix in line 4 of model.py where trying to import from deephcd but need to drop the deephcd part
#fix in split_dataset where List[float,float] is invalid
#line 498 in trainer only takes 1 argument for int as well
#import Trainer not fit
#remove A from get_efficient_batch_data call in fit
#bandaid solution for resolutions in forward function making it the size of all_A
#simuilarily make bandaid for lambda making it a list of the size of all_A
#print logger was taking in multiple arguments so reduce that to one
#added condition for test data for early stopping
#remove device argument in output
#must have directory already creted