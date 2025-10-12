import networkx as nx
import numpy as np
from random import randint as rd    



def add_edges_to_subgraph(G, nodes_to_add, edges_to_add, weighted_graph = True, 
                       weighting = (0.1, 0.5)):
    
    low, high = weighting
    if nodes_to_add is not None:
        G.add_nodes_from(nodes_to_add)
    for edge in edges_to_add:
        p_node, c_node = edge
        if weighted_graph:
            G.add_edge(p_node, c_node,  weight = np.round(np.random.uniform(low, high),2))
        else:
            G.add_edge(p_node, c_node, weight = 1)
    return G.copy()




def add_edges_to_fullgraph(G, nodes_to_add, edges_to_add):
    
    if nodes_to_add is not None:
        G.add_nodes_from(nodes_to_add)
    for edge in edges_to_add:
        p_node, c_node, weight = edge
        G.add_edge(p_node, c_node, weight = weight)
    return G.copy()




def make_dag(graph):
    # Step 1: Replace bidirectional edges with a single directed edge
    edges = list(graph.edges)
    for u, v in edges:
        if graph.has_edge(v,u) and graph.has_edge(u,v):
            # Remove both edges to eliminate bidirectionality
            graph.remove_edge(u, v)
            graph.remove_edge(v, u)
            # Add a single directed edge (choose one direction, e.g., u -> v)
            graph.add_edge(u, v)

    # Step 2: Remove cycles by flipping edges
    print('\n resolving cycles...')
    while not nx.is_directed_acyclic_graph(graph):
        # Find a cycle
        try:
            cycle = nx.find_cycle(graph, orientation='original')
        except nx.NetworkXNoCycle:
            break
        
        # Flip an edge in the cycle
        u, v, _ = cycle[0]
        graph.remove_edge(u, v)
        graph.add_edge(v, u)
    
    return graph






def hierachical_graph(top_graph, subgraph_node_number, subgraph_type, as_weighted = True, 
                      degree=3, connection_prob_within=0.05, connection_prob_between = 0.01,
                      sub_graph_prob=0.01, weight_b = (0.1, 0.3), weight_w = (0.4, 0.8), mixed = False, 
                      force_connections = False, seed = None):
    """
    Constructs a hierarchical directed graph using subgraphs of specified types and properties, 
    based on a given top-level graph structure. The function supports multiple subgraph types and 
    allows for mixed or uniform subgraph configurations.

    Args:
        top_graph (networkx.Graph): The base graph defining the hierarchy among subgraphs.
        subgraph_node_number (tuple): Range (min, max) for the number of nodes in each subgraph.
        subgraph_type (str): Type of subgraph to be used ('small world', 'random graph', or 'scale free').
        as_weighted (bool, optional): If True, assigns weights to the edges in the subgraphs. Defaults to True.
        degree (int, optional): Degree parameter for subgraphs where applicable (e.g., small world). Defaults to 3.
        connection_prob_within (float, optional): Probability of connecting nodes within the same community. Defaults to 0.05.
        connection_prob_between (float, optional): Probability of connecting nodes between different communities. Defaults to 0.01.
        sub_graph_prob (float, optional): Probability of connections within subgraphs. Defaults to 0.01.
        weight_b (tuple, optional): Range (min, max) for weights between subgraph connections. Defaults to (0.1, 0.3).
        weight_w (tuple, optional): Range (min, max) for weights within subgraph connections. Defaults to (0.4, 0.8).
        mixed (bool, optional): If True, uses a mix of subgraph types. Otherwise, uses a uniform type. Defaults to False.
        force_connections (bool, optional): If True, ensures at least one connection between communities even if probability conditions fail. Defaults to False.
        seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - full_graph (networkx.DiGraph): The complete hierarchical graph constructed.
            - subgraphs (list): A list of subgraphs used in constructing the full graph.

    This function generates a hierarchical graph structure by populating a base graph with subgraphs
    defined by the specified parameters. It handles different configurations for subgraphs, supports
    both directed and weighted connections, and allows for probabilistic inter-community links.
    """
    subgraphs, node_list = [], []
    full_graph = nx.DiGraph()
    full_graph_node_list = {}

    # if no seed is select a seed at random
    if not seed:
        seed = rd(100, 500)
        print(f'Using random seed {seed}')

    #for mixed topologies
    if (mixed == True and (subgraph_type == 'small world' or subgraph_type == 'random graph' or subgraph_type == 'scale free')):
        top_node_length =  len(list(top_graph.nodes())) ; g_index  = top_node_length // 3
        top_node_g1 = list(top_graph.nodes())[:g_index]
        top_node_g2 = list(top_graph.nodes())[g_index:(g_index + g_index)]
        top_node_g3 = list(top_graph.nodes())[(g_index + g_index):]
        
        #small world
        for top_node_g1 in top_node_g1: 
            subgraph_sm = nx.watts_strogatz_graph(n=rd(subgraph_node_number[0], 
                                                     subgraph_node_number[1]), 
                                                  k=degree, 
                                                  p=sub_graph_prob/np.mean(subgraph_node_number))
            
            #subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph_sm.edges() if u!=v]))
            G = add_edges_to_subgraph(nx.DiGraph(), subgraph_sm.nodes, subgraph_sm.edges, 
                                       weighted_graph = as_weighted, weighting = weight_w)
            
            subgraphs.append(G)
            node_list.append(top_node_g1)
            print('small world',subgraph_sm.nodes())
            print(subgraph_sm.edges())
        
        #random graph
        for top_node_g2 in top_node_g2:
            subgraph_rd = nx.gnp_random_graph(n=rd(subgraph_node_number[0], 
                                                     subgraph_node_number[1]),
                                                  p=sub_graph_prob,
                                                  directed=True)
            #subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph_rd.edges() if u<v]))
            #check for cyclicity
            if not nx.is_directed_acyclic_graph(subgraph_rd):
                G = make_dag(subgraph_rd)
                    
            G = add_edges_to_subgraph(nx.DiGraph(), subgraph_rd.nodes, subgraph_rd.edges, 
                                       weighted_graph = as_weighted, weighting = weight_w)
                    
            subgraphs.append(G)
            node_list.append(top_node_g2)
            print('random',subgraph_rd.nodes())
            print(subgraph_rd.edges())
            
            
        #scale free
        for top_node_g3 in top_node_g3: 
            n = rd(subgraph_node_number[0], subgraph_node_number[1]); m = rd(2,(n-1))
            subgraph_sf = nx.barabasi_albert_graph(n,m)
            
            #subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph_sf.edges() if u!=v]))
            G = add_edges_to_subgraph(nx.DiGraph(), subgraph_sf.nodes, subgraph_sf.edges, 
                                       weighted_graph = as_weighted, weighting = weight_w)
            
            subgraphs.append(G)
            node_list.append(top_node_g3)
            print('scale free',subgraph_sf.nodes())
            print(subgraph_sf.edges())
        
        for i in range(0,len(subgraphs)):
            print(i)
            print(subgraphs[i])
            print(list(nx.topological_sort(subgraphs[i])))
        
        print("Mixed subgraghs used")
        
    else :
        
        # for uniform topologies 
        for index, topgraph_node in enumerate(list(top_graph.nodes)):
            
            #small world
            if (subgraph_type == 'small world' and (mixed == False or mixed is None)):
                subgraph = nx.watts_strogatz_graph(n=rd(subgraph_node_number[0], 
                                                      subgraph_node_number[1]), 
                                                   k=degree, 
                                                   p=sub_graph_prob/np.mean(subgraph_node_number),
                                                   seed = seed)
                
                
                G = add_edges_to_subgraph(nx.DiGraph(), subgraph.nodes, subgraph.edges, 
                                       weighted_graph = as_weighted, weighting = weight_w)
                subgraphs.append(G)
            
            #random graph
            elif (subgraph_type == 'random graph' and (mixed == False or mixed is None)):
                
                subgraph = nx.gnp_random_graph(n = rd(subgraph_node_number[0], 
                                                        subgraph_node_number[1]), 
                                                p = sub_graph_prob,
                                                directed = True)
                
                if not nx.is_directed_acyclic_graph(subgraph):
                    G = make_dag(subgraph)
                    
                G = add_edges_to_subgraph(nx.DiGraph(), subgraph.nodes, subgraph.edges, 
                                       weighted_graph = as_weighted, weighting = weight_w)
                
                subgraphs.append(G)
                
            #Scale free subgraph
            elif (subgraph_type == 'scale free' and (mixed == False or mixed is None)):
                n = rd(subgraph_node_number[0], subgraph_node_number[1]); m = rd(2,(n-1))
                subgraph = nx.barabasi_albert_graph(n,m)
                G = add_edges_to_subgraph(nx.DiGraph(), subgraph.nodes, subgraph.edges, 
                                       weighted_graph = as_weighted, weighting = weight_w)
                subgraphs.append(G)

            node_list.append(topgraph_node)

        print(subgraph_type, "subgraphs used")
        
    # generate full graph based on top graph
    # creates a directed DiGraph for each community corresponding nodes in the layer
    # above
    for index, topgraph_node in enumerate(node_list):
        # add nodes and edges to full_graph
        nodes = [str(topgraph_node) + '_' + str(node) for node in list(subgraphs[index].nodes)]
        edges = [(str(topgraph_node) + '_' + str(u), 
                  str(topgraph_node) + '_' + str(v), 
                  subgraphs[index].get_edge_data(u, v)['weight']) for (u,v) in subgraphs[index].edges]
        full_graph = add_edges_to_fullgraph(full_graph, 
                                            nodes_to_add = nodes, 
                                            edges_to_add = edges)
        full_graph_node_list[topgraph_node] = nodes
        
    print(f'{top_graph.edges}')
    # add connections between sub-graphs
    for p_graph, c_graph in top_graph.edges:
        
        p_graph_name = str(p_graph).split('_')
        c_graph_name = str(c_graph).split('_')
        
        if len(p_graph_name) == 2:
            p_community, p_node_name = p_graph_name
            c_community, c_node_name = c_graph_name
        else:
            p_community = p_graph
            c_community = c_graph
        
        # get node lists in graph p and c, where edge p->c exists in top-graph
        p_list, c_list = full_graph_node_list[p_graph], full_graph_node_list[c_graph]
        # all possible connections between communities
        possible_edges = [(i, j) for i in p_list for j in c_list]
        np.random.shuffle(possible_edges)
        num_possible = len(possible_edges)
        #generate edges
        if p_community == c_community:
            c_prob = connection_prob_within
        else: 
            c_prob = connection_prob_between
            
        #get edge indices
        which_edges=[i<=c_prob for i in np.random.uniform(size = num_possible).tolist()]
        
        #handle forced connections (select one edge at random from possible edges)
        if sum(which_edges) == 0 and force_connections:
            which_edges_idx = np.array([np.random.randint(num_possible)])
        else:
            which_edges_idx = np.arange(num_possible)[which_edges]
            
        #add edges and print which edges were added to user
        edges_to_add = [possible_edges[x] for x in which_edges_idx]
        print('cprob: ',c_prob)
        print('num_possible: ',num_possible)
        num_true = 0
        for i in range(len(which_edges)):
            if which_edges[i] == True:
                num_true+=1
        print('which_edges are true:', num_true)

        print(f'Adding {len(edges_to_add)} edges between community {p_graph} and community {c_graph}: \n {edges_to_add}')
       
        full_graph = add_edges_to_subgraph(full_graph, 
                                           nodes_to_add = None, 
                                           edges_to_add = edges_to_add, 
                                           weighted_graph = as_weighted, 
                                           weighting = weight_b)
    
    return full_graph, subgraphs





def generate_pseudo_expression(topological_order, adjacency_matrix, 
                               number_of_invididuals, free_mean=0, std=0.5,
                               common_distribution = True):
    
    """
    Generates pseudo-expression data for a directed acyclic graph (DAG) based on 
    the given topological order and adjacency matrix. This function simulates 
    expression levels for nodes, where the expression level of each node is 
    influenced by its parent nodes.

    Args:
        topological_order (list): A list of nodes in topological order, representing the DAG.
        adjacency_matrix (numpy.ndarray): A binary adjacency matrix indicating the presence of edges 
            between nodes (1 for an edge from row node to column node, 0 otherwise).
        number_of_invididuals (int): The number of individual samples for which to generate expression data.
        free_mean (float, optional): The mean of the normal distribution for generating expression levels 
            for nodes without parents. Defaults to 0.
        std (float, optional): The standard deviation of the normal distribution used for generating 
            expression levels. Defaults to 0.5.
        common_distribution (bool, optional): If True, uses a common distribution with mean `free_mean` 
            for all origin nodes. If False, uses the index of the node as the mean for each origin node. 
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - pseudo_expression (numpy.ndarray): A matrix of generated pseudo-expression data, 
              with shape (number_of_nodes, number_of_invididuals).
            - origin_nodes (list): A list of tuples (index, node) for nodes without parent nodes in the DAG.

    This function iterates over each node in topological order, determining whether it has parents 
    based on the adjacency matrix. For nodes without parents, it generates expression data from 
    a normal distribution. For nodes with parents, it generates data based on the mean expression 
    level of the parent nodes.
    """
    
    N = len(topological_order)
    pseudo_expression = np.zeros((N, number_of_invididuals))
    origin_nodes = []
    for index, node in enumerate(topological_order):
        if np.sum(adjacency_matrix[:,index]) == 0:
            origin_nodes.append([index,node])
            if common_distribution == True:
                pseudo_expression[index,:] = np.random.normal(free_mean, std,
                                                          size = number_of_invididuals)
            else:
                pseudo_expression[index,:] = np.random.normal(index,
                                                              std,
                                                              size = number_of_invididuals)
        else:
            parents_idx = [i for i in np.arange(N) if adjacency_matrix[i, index]==1]
            parents_loc = pseudo_expression[parents_idx, :].mean(axis = 0)
            pseudo_expression[index, :] = np.random.normal(parents_loc, std) 
    return pseudo_expression, origin_nodes





def generate_pseudo_expression_weighted(topological_order, adjacency_matrix, 
                                        number_of_invididuals, free_mean=0, std=0.5,
                                        common_distribution = True):
    
    
    """
    Generates weighted pseudo-expression data for a directed acyclic graph (DAG) based on 
    the given topological order and weighted adjacency matrix. This function simulates 
    expression levels for nodes, where the expression level of each node is influenced by 
    its parent nodes, accounting for edge weights.

    Args:
        topological_order (list): A list of nodes in topological order, representing the DAG.
        adjacency_matrix (numpy.ndarray): A weighted adjacency matrix indicating the strength 
            of edges between nodes (non-zero values indicate an edge from row node to column node).
        number_of_invididuals (int): The number of individual samples for which to generate expression data.
        free_mean (float, optional): The mean of the normal distribution for generating expression levels 
            for nodes without parents. Defaults to 0.
        std (float, optional): The standard deviation of the normal distribution used for generating 
            expression levels. Defaults to 0.5.
        common_distribution (bool, optional): If True, uses a common distribution with mean `free_mean` 
            for all origin nodes. If False, uses the index of the node as the mean for each origin node. 
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - pseudo_expression (numpy.ndarray): A matrix of generated pseudo-expression data, 
              with shape (number_of_nodes, number_of_invididuals).
            - origin_nodes (list): A list of tuples (index, node) for nodes without parent nodes in the DAG.

    This function iterates over each node in topological order, determining whether it has parents 
    based on the weighted adjacency matrix. For nodes without parents, it generates expression data 
    from a normal distribution. For nodes with parents, it generates data based on a weighted average 
    of the expression levels of the parent nodes, using the weights from the adjacency matrix.
    """
    
    N = len(topological_order)
    pseudo_expression = np.zeros((N, number_of_invididuals))
    origin_nodes = []
    for index, node in enumerate(topological_order):
        if np.sum(adjacency_matrix[:,index]) == 0:
            origin_nodes.append([index,node])
            if common_distribution == True:
                pseudo_expression[index,:] = np.random.normal(free_mean, std,
                                                          size = number_of_invididuals)
            else:
                pseudo_expression[index,:] = np.random.normal(index,
                                                              std,
                                                              size = number_of_invididuals)
        else:
            parents_idx = [i for i in list(np.arange(N)) if adjacency_matrix[i, index] != 0]
            weights = adjacency_matrix[parents_idx, index].reshape(len(parents_idx), 1)
            weights = adjacency_matrix[parents_idx, index]
            parents_loc = np.multiply(pseudo_expression[parents_idx, :].transpose(),weights).sum(axis = 1).reshape(1, number_of_invididuals)
            parents_loc = np.matmul(pseudo_expression[parents_idx, :].transpose(), weights).reshape(1, number_of_invididuals)
            pseudo_expression[index, :] = np.random.normal(parents_loc, std) 
    return pseudo_expression, origin_nodes



def same_cluster(s1, s2):
    l1 = s1.split('_')
    l2 = s2.split('_')
    if l1[0] == l2[0] and l1[1] == l2[1] and l1[2] != l2[2]:
        return True
    else:
        return False