# Import packages
#%matplotlib inline
import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import pandas as pd
import time
from igraph import *
import csv


# Generate Graph , nodes=10, edges=30 and export to csv
G = Graph.Erdos_Renyi(n=10000,m=30000,directed=True)
# Plot Graph
G.es["color"], G.vs["color"], G.vs["label"] = "#B3CDE3", "#FBB4AE", ""
#plot(G, bbox = (300, 300),margin = 11,layout = G.layout("kk"))
d=G.get_edgelist()
with open('d.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
      
    #write.writerow(fields)
    write.writerows(d)
    
    
#Read csv as Graph object
df=pd.read_csv("d.csv", header=None)


source=df[0]
s=list(source)
target=df[1]
t=target.tolist()
g1 = Graph(directed=True)
g1.add_vertices(range(10000))
g1.add_edges(zip(s,t))
# Plot graph
g1.vs["label"], g1.es["color"], g1.vs["color"] = range(1000), "#B3CDE3", "#FBB4AE"
#png("my_plot.png", 600, 600)
#dev.off()
#plot(g1,bbox = (200,200),margin = 20, layout = g1.layout("kk"),)
#plot(g1, "g4.png", dpi=300)


#convert df by adding compatibility factor
df['compf'] = np.random.randint(0,2, size=len(df))


#removing entries where compatibility = 0
df1= df[df['compf'] != 0]


#regenerating graph
source=df1[0]
s=list(source)
target=df1[1]
t=target.tolist()
g2 = Graph(directed=True)
g2.add_vertices(range(10000))
g2.add_edges(zip(s,t))
g2.vs.select(_degree=0).delete()
# Plot graph
#g2.vs["label"], g2.es["color"], g2.vs["color"] = range(1000), "#B3CDE3", "#FBB4AE"
#png("my_plot.png", 600, 600)
#dev.off()
#plot(g2,bbox = (200,200),margin = 20,layout = g2.layout("kk"),)
#plot(g2, "g3_modified.png", dpi=300)



def IC(g,S,p=0.5,mc=1000):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """
    
    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(mc):
        
        # Simulate propagation process      
        new_active, A = S[:], S[:]
        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:
                
                # Determine neighbors that become infected
                np.random.seed(i)
                success = np.random.uniform(0,1,len(g.neighbors(node,mode="out"))) < p
                new_ones += list(np.extract(success, g.neighbors(node,mode="out")))

            new_active = list(set(new_ones) - set(A))
            
            # Add newly activated nodes to the set of activated nodes
            A += new_active
            
        spread.append(len(A))
        print(A)
        
    return(np.mean(spread))



def celf(g,k,p=0.1,mc=1000):  
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
      
    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    
    # Calculate the first iteration sorted list
    start_time = time.time() 
    marg_gain = [IC(g,[node],p,mc) for node in range(g.vcount())]

    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(range(g.vcount()),marg_gain), key=lambda x: x[1],reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [g.vcount()], [time.time()-start_time]
    
    # --------------------
    # Find the next k-1 nodes using the list-sorting procedure
    # --------------------
    
    for _ in range(k-1):    

        check, node_lookup = False, 0
        
        while not check:
            
            # Count the number of times the spread is computed
            node_lookup += 1
            
            # Recalculate spread of top node
            current = Q[0][0]
            
            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current,IC(g,S+[current],p,mc) - spread)

            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse = True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        # Remove the selected node from the list
        Q = Q[1:]

    return(S,SPREAD,timelapse,LOOKUPS)



def greedy(g,k,p=0.1,mc=1000):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    S, spread, timelapse, start_time = [], [], [], time.time()
    
    # Find k nodes with largest marginal gain
    for _ in range(k):

        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0
        for j in set(range(g.vcount()))-set(S):

            # Get the spread
            s = IC(g,S + [j],p,mc)

            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j

        # Add the selected node to the seed set
        S.append(node)
        print(S)
        
        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)

    return(S,spread,timelapse)


print("running on original graph")
# Run algorithms with unmodified graph
celf_output1   = celf(g1,10,p = 0.1,mc = 1000)
greedy_output1 = greedy(g1,10,p = 0.1,mc = 1000)


print("running on modified graph")
# Run algorithms with modified graph
celf_output2   = celf(g2,10,p = 0.1,mc = 1000)
greedy_output2 = greedy(g2,10,p = 0.1,mc = 1000)



#1st plot
# Plot settings
plt.rcParams['figure.figsize'] = (9,6)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
# Plot Computation Time
plt.plot(range(1,len(greedy_output1[2])+1),greedy_output1[2],label="Greedy",color="green")
plt.plot(range(1,len(celf_output1[2])+1),celf_output1[2],label="CELF",color="yellow")
plt.plot(range(1,len(greedy_output2[2])+1),greedy_output2[2],label="Context Aware Greedy",color="red")
plt.plot(range(1,len(celf_output2[2])+1),celf_output2[2],label="Context Aware CELF",color="blue")
plt.ylabel('Computation Time (Seconds)'); plt.xlabel('Size of Seed Set')
plt.title('Computation Time'); plt.legend(loc=2);
plt.tight_layout()
plt.savefig("g4i.jpg", dpi=300)


#2nd plot
plt.plot(range(1,len(greedy_output1[1])+1),greedy_output1[1],label="Greedy",color="green")
plt.plot(range(1,len(celf_output1[1])+1),celf_output1[1],label="CELF",color="yellow")
plt.plot(range(1,len(greedy_output2[1])+1),greedy_output2[1],label="Context Aware Greedy",color="red")
plt.plot(range(1,len(celf_output2[1])+1),celf_output2[1],label="Context Aware CELF",color="blue")
plt.xlabel('Size of Seed Set'); plt.ylabel('Expected Spread')
plt.title('Expected Spread'); plt.legend(loc=2);
plt.tight_layout()
plt.savefig("g4ii.jpg", dpi=300)

