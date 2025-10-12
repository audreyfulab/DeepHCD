import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import networkx as nx
import numpy as np

# a simple function to plot the loss curves during training
#----------------------------------------------------------------
def plot_loss(epoch, layers, train_loss_history, test_loss_history, true_losses = None, path='path/to/file', save = True):
    
    
    
    total_train = [i['Total Loss'] for i in train_loss_history]
    total_test =[i['Total Loss'] for i in test_loss_history]
    
    recon_A_train = [i['A Reconstruction'] for i in train_loss_history]
    recon_A_test = [i['A Reconstruction'] for i in test_loss_history]
    
    recon_X_train = [i['X Reconstruction'] for i in train_loss_history]
    recon_X_test = [i['X Reconstruction'] for i in test_loss_history]
    
    mod_train = [i['Modularity'] for i in train_loss_history]
    mod_test = [i['Modularity'] for i in test_loss_history]
    
    clust_train = [i['Clustering'] for i in train_loss_history]
    clust_test = [i['Clustering'] for i in test_loss_history]
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(3,2, figsize=(12,10))
    #total loss
    ln1, = ax1[0].plot(range(0, epoch+1), total_train, label = 'Train')
    ln2, = ax1[0].plot(range(0, epoch+1), total_test, linestyle = 'dashed', label = 'Test')
    ax1[0].set_xlabel('Training Epochs')
    ax1[0].set_ylabel('Total Loss')
    #reconstruction of graph adjacency
    ln3, = ax1[1].plot(range(0, epoch+1), recon_A_train, label = 'Train')
    ln4, = ax1[1].plot(range(0, epoch+1), recon_A_test, linestyle = 'dashed',  label = 'Test')
    ax1[1].set_xlabel('Training Epochs')
    ax1[1].set_ylabel('Graph Reconstruction Loss')
    #reconstruction of node attributes
    ln5, = ax2[0].plot(range(0, epoch+1), recon_X_train, label = 'Train')
    ln6, = ax2[0].plot(range(0, epoch+1), recon_X_test, linestyle = 'dashed', label = 'Test')
    ax2[0].set_xlabel('Training Epochs')
    ax2[0].set_ylabel('Gamma * Attribute Reconstruction Loss')
    #community loss using modularity
    lines1a, lines1b = ax2[1].plot(range(0, epoch+1), np.array(mod_train), label = ['train top', 'train middle'])
    lines2a, lines2b = ax2[1].plot(range(0, epoch+1), np.array(mod_test), label = ['test top', 'test middle'], linestyle = 'dashed')
    ax2[1].set_xlabel('Training Epochs')
    ax2[1].set_ylabel('Delta * Modularity')
    #community loss using kmeans
    lines3a, lines3b = ax3[0].plot(range(0, epoch+1), np.array(clust_train), label = ['train top', 'train middle'])
    lines4a, lines4b = ax3[0].plot(range(0, epoch+1), np.array(clust_test), label = ['test top', 'test middle'], linestyle ='dashed')
    if true_losses:
        ax3[0].axhline(y=true_losses[0], color='black', linestyle='dotted', linewidth=2)
        ax3[0].axhline(y=true_losses[1], color='black', linestyle='dotted', linewidth=2)
    ax3[0].set_xlabel('Training Epochs')
    ax3[0].set_ylabel('Lambda * Clustering Loss')
    ax3[1].axis('off')
    
    ax1[0].legend(handles = [ln1, ln2], loc = 'lower right')
    ax1[1].legend(handles = [ln3, ln4], loc = 'lower right')
    ax2[0].legend(handles = [ln5, ln6], loc = 'lower right')
    
    ax3[1].legend(handles = [lines3a, lines4a, lines3b, lines4b], bbox_to_anchor=(0.5, 0.5), loc = 'lower right')
    
    if save == True:
        fig.savefig(path+'training_loss_curve_epoch_'+str(epoch+1)+'.pdf')
    plt.close(fig)
    




# a simple function for plotting the performance curves during training
#----------------------------------------------------------------
def plot_perf(update_time, performance_hist, valid_hist, epoch, path='path/to/file', save=True):
    # Skip plotting if no performance history exists
    if not performance_hist or all(p is None for p in performance_hist):
        print("Skipping plotting: No performance data available.")
        return

    # Filter out None values (evaluation was skipped in some epochs)
    valid_perf_hist = [p for p in performance_hist if p is not None]
    if not valid_perf_hist:
        return

    # Use the last valid performance entry for layers/titles
    layers = len(valid_perf_hist[-1])
    titles = ['Top Layer', 'Middle Layer'][:layers]  # Adjust titles based on layers

    fig, ax = plt.subplots(1, layers, figsize=(12, 10))
    for i in range(layers):
        layer_hist = [j[i] for j in valid_perf_hist if j is not None]
        
        # Plot training metrics
        ax[i].plot(range(len(layer_hist)), [x[0] for x in layer_hist], label='Homogeneity')
        ax[i].plot(range(len(layer_hist)), [x[3] for x in layer_hist], label='ARI')
        ax[i].set_title(titles[i])
        ax[i].legend()

    # Handle validation data if exists
    if valid_hist and not all(v is None for v in valid_hist):
        valid_perf = [v for v in valid_hist if v is not None]
        for i in range(layers):
            ax[i].plot(range(len(valid_perf)), [x[i][0] for x in valid_perf], '--', label='Val Homogeneity')
            ax[i].plot(range(len(valid_perf)), [x[i][3] for x in valid_perf], '--', label='Val ARI')
            ax[i].legend()

    if save:
        plt.savefig(f'{path}/performance_epoch_{epoch}.png')
    plt.close('all')
            
            
            
#A simple wrapper to plot and save the networkx graph
#----------------------------------------------------------------
def plot_nodes(A, labels, path, node_size = 5, font_size = 10, add_labels = False,
               save = True, **kwargs):
    fig, ax = plt.subplots()
    G = nx.from_numpy_array(A)
    if add_labels == True:
        clust_labels = {list(G.nodes)[i]: labels.tolist()[i] for i in range(len(labels))}
        nx.draw_networkx(G, node_color = labels, 
                         pos = nx.spring_layout(G, seed = 123),
                         labels = clust_labels,
                         font_size = font_size,
                         node_size = node_size,
                         cmap = 'plasma', **kwargs)
    else:
        nx.draw_networkx(G, node_color = labels, 
                         ax = ax, 
                         pos = nx.spring_layout(G, seed = 123),
                         font_size = font_size,
                         node_size = node_size, 
                         with_labels = False,
                         cmap = 'plasma', **kwargs)
    if save == True:    
        fig.savefig(path+'.png')
    plt.close(fig)
    
  
    
  
    
  
    
#A simple wrapper to plot and save the adjacency heatmap
#----------------------------------------------------------------
def plot_adj(A, path, **kwargs):
    fig, ax = plt.subplots()
    sbn.heatmap(A, ax = ax, **kwargs)
    fig.savefig(path+'.png', dpi = 300)
    
    
    
    
    
    
    
    
    
    
    
    
    
#a simple function to plot the clustering heatmaps
#---------------------------------------------------------------- 
def plot_clust_heatmaps(A, A_pred, X, X_pred, true_labels, pred_labels, layers, epoch, save_plot = True, sp = ''):
    with plt.style.context('default'):
        fig1, ax1 = plt.subplots(1, 2, figsize=(12, 10))
        sbn.heatmap(A_pred.cpu().detach().numpy(), ax=ax1[0])
        sbn.heatmap(A.cpu().detach().numpy(), ax=ax1[1])
        ax1[0].set_title(f'Reconstructed Adjacency At Epoch {epoch}')
        ax1[1].set_title('Input Adjacency Matrix')
        
    fig1, ax1 = plt.subplots(1,2, figsize=(12,10))
    sbn.heatmap(A_pred.cpu().detach().numpy(), ax = ax1[0])
    sbn.heatmap(A.cpu().detach().numpy(), ax = ax1[1])
    ax1[0].set_title(f'Reconstructed Adjacency At Epoch {epoch}')
    ax1[1].set_title('Input Adjacency Matrix')
    
    
    fig11, ax11 = plt.subplots(1,2, figsize=(12,10))
    sbn.heatmap(X_pred.cpu().detach().numpy(), ax = ax11[0])
    ax11[0].set_title(f'Reconstructed Attributes At Epoch {epoch}')
    sbn.heatmap(X.cpu().detach().numpy(), ax = ax11[1])
    ax11[1].set_title('Input Attributes')
    
    
    
    fig2, ax2 = plt.subplots(1,2, figsize=(12,10)) 
        
    if true_labels:
        first_layer = pd.DataFrame(np.vstack((pred_labels[0], true_labels[0])).T,
                                   columns = ['Predicted_Top','Truth_Top'])
        
        if layers == 3:
            second_layer = pd.DataFrame(np.vstack((pred_labels[1], true_labels[1])).T,
                                            columns = ['Predicted_Middle','Truth_Middle'])
    else:
        first_layer = pd.DataFrame(np.array(pred_labels[0]).T, columns=['Predicted_Top'])
        
        if layers == 3:
            second_layer = pd.DataFrame(np.array(pred_labels[1]).T, columns=['Predicted_Middle'])
    
    sbn.heatmap(first_layer, ax = ax2[0])
    ax2[0].set_title(f'Predictions (Top) at epoch {epoch}')
    
    if layers == 3:
        sbn.heatmap(second_layer, ax = ax2[1])
        ax2[1].set_title(f'Predictions (Middle) at epoch {epoch}')
        
    if save_plot == True:
        fig1.savefig(sp+'epoch_'+str(epoch)+'_Adjacency_maps.png', dpi = 300)
        fig2.savefig(sp+'epoch_'+str(epoch)+'_heatmaps.png', dpi = 300) 
        
    plt.close('all')
