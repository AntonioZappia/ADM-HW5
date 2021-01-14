import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from random import choice
from collections import deque
import networkx as nx
import pandas as pd
import operator
from tqdm import tqdm

def read_graph_csv(csv):
    graph = pd.read_csv(csv, sep='\t', usecols =["0","1"]) #Read file csv with separator \t
    graph.columns = ["Source","Target"] #Rename our columns
    return graph

def read_page_names(txt):
    data_name = pd.read_csv(txt,sep="\t", names = ["Page Names"]) #Read file csv with separator \t
    node =[] #Create a empty list for node
    page= [] ##Create a empty list for node
    for i in (data_name["Page Names"]): #We extract in each row from single column node and page
        node.append(i.split()[0]) #Split the single column and take the first element (NODE)
        page.append(" ".join(i.split()[1:])) #From the second element we have the name of page
    data_name["Node"] = node #Create new column for Node
    data_name["Pages"] = page #Create new column for pages
    data_name = data_name.drop(["Page Names"], axis=1).set_index("Node") #Drop our original column and set our index in colun node
    return data_name

def read_top_categories(txt):
    data_cat = pd.read_csv(txt,sep=';', names = ["Category","PageList"]) #Read file csv with separator ";"
    data_cat['Category'] = data_cat['Category'].map(lambda x: x.lstrip('Category:').rstrip('aAbBcC')) #modify the Category column by removing the initial part of the string "Category:" for each row
    data_cat['PageList'] = data_cat['PageList'].map(lambda x: x.split()) #modify the PageList column by splitting our string of pageList in a list
    data_cat = data_cat[(data_cat['PageList'].map(len) >= 5000) & (data_cat['PageList'].map(len) <= 30000)].reset_index() #Remove all Category with a number of Page lower than 5000 and higher than 30000
    data_cat = data_cat.drop("index", axis = 1)
    return data_cat

def random_dictionary(df):
    dictionary = {} #Empty dictionary
    for i in tqdm(range(len(df))): #For each row in df_article dataframe
        if df.loc[i,"Articles"] not in dictionary: #If Article not as dictionary key
            dictionary[df.loc[i,"Articles"]] = [] #Put this article like key and create for this key create an empty list
        dictionary[df.loc[i,"Articles"]].append(df.loc[i,"Category"])#Append for each key (Article) list of category where we found this Article
    for keys in dictionary: #For each key in dictionary
        new_value = random.choice(dictionary[keys]) #We decide with random a value of each keys and save this value like new_value
        del dictionary[keys][:] #Remove all element in a list linked to specific keys
        dictionary[keys] = new_value #Put for this key the new random_value
    return dictionary
    
def reversed_dictionary(dictionary):
    reversed_dict = {} #New dictionary to put like keys the category and for each category define all Articles
    for key, value in dictionary.items(): #For each tuple in "dictionary" containing key and value
        reversed_dict.setdefault(value, []) #New key of reversed_dict equal to value in dictionary
        reversed_dict[value].append(key) #Append value for each key in reversed_dict that are keys in "dictionary"
    new_dat = pd.DataFrame(reversed_dict.items(), columns=['Category', 'PageList']) #Obtain dataframe from dictionary
    return new_dat

def is_directed(df):
    #creating the inverted graph
    graph_ = df.copy()
    graph_inverted = graph_[graph_.columns[::-1]]
    #converting the dataframes in a list of edges
    set_1 = df.values.tolist()
    set_2 = graph_inverted.values.tolist()
    intersect = [value for value in set_1 if value in set_2]
    if (intersect == set_1): # is the intersection equal to list of edge?
        return print('The graph is not directed')
    else:
        return print('The graph is directed')


def create_graph(df):
    #empty graph G
    G = nx.DiGraph() #Create from networkx an empty directed Graph
    sources = list(df['Source']) #List of all nodes that are source in a graph
    targets = list(df['Target'])#List of all nodes that are Target in a graph

    #building G = (V,E)
    for i in range(len(df)): #For each row in graph dataset
        if sources[i] not in G.nodes: #If source linked to i-th row not in G.nodes
            G.add_node(sources[i]) #Add sources[i] like new nodes
        if targets[i] not in G.nodes:#If target linked to i-th row not in G.nodes
            G.add_node(targets[i]) ##Add targets[i] like new nodes
        G.add_edge(sources[i], targets[i], weight=1) #Create edge from i-th sources and i-th targets wit a weight equal o 1

    return G

def number_articles(df):
    sources = set(df['Source']) #Set of all nodes that are in Source column (Drop the duplicate)
    targets = set(df['Target']) #Set of all nodes that are in Target column (Drop the duplicate)
    return len(sources.union(targets)) #Compute the len of new set obtained from the union of sources and targets sets


def number_hyperlinks(df):
    return len(df) #Compute the number of rows in Dataframe containing Edge List because number of hyperlinks equal to number of edges that are equal to number of rows in Graph Dataframe

def average_links(graph):
    somma = 0 #We initialized somma variable to zero
    for node in graph.nodes: #For each nodes in graph
        somma += nx.degree(graph)[node] #We increase the somma variable with the degree of specific node in Graph
    return somma/len(graph.nodes) #Average links for each node equal to ratio between the sum and the number of nodes

def graph_density(nodes,edges):
    density = edges/(nodes*(nodes-1)) #Graph density equal to ratio between number of edges and twice the binomial coefficient of the number of nodes over 2
    return density

def comparison_edges(nodes,edges): #We want to prove that our graph is sparse to do this we compute the number of max_edges in our Graph and number of real eadges
    max_edges = nodes*(nodes-1)
    return print("The value of max edges is", max_edges, "while the value of edges is equals", edges)

def plot_degree_dist(graph):
    degrees = [graph.degree(node) for node in graph.nodes()] #We compute for each node the degree in our Graph
    plt.hist(degrees, bins=[0, 10, 20, 30, 40, 50, 55,60]) #We plot degree's node distribution with an histogram
    plt.show()

def clicks(graph, start_node, depth):
    visited = [start_node] #the starting node is the first visited node
    for i in range(depth): #until level d 
        all_visited =[] # all the nodes visited
        for node in visited:
            for nbr in graph[node]: #for all the neighborhood  of node
                all_visited.append(nbr)  #save the neighborhood in the list 
        visited = all_visited#saving all the neighborhood     
    return set(all_visited) #returning a set to avoid ripetitions

def category(df):
    nodes = list() #Create an empty list  
    C = input("Enter your Category: ") #We ask as input a Category
    for i in range(len(df)): #For each row in our dataframe containing category and page lsit
        if df.loc[i,'Category'] == C: #If the category that we put in input is equal to category in i-th row
            nodes = df.loc[i,"PageList"] #We increase our nodes list with all ele
    if not nodes: #If nodes list is empty
        print("No Category") #Category not appear in our dataframe
    else:
        return list(map(int, nodes)) #Map every element in list nodes to int

def degree_centrality(G,nodes): #Find the nodes with best centrality
    dictionary = {} #Create an empty dictionary
    tot_graph = nx.degree_centrality(G) #Define with networkx function the degree centrality for each node in graph. We obtain a dictionary with keys the node and value the centrality
    for keys,value in tot_graph.items(): #For each tuple (key,value) in dictioanry created 
        if int(keys) in nodes: #If the keys in list of pages for specific category
            dictionary.update({keys: value}) #I put in the empty dictionary the tuple key,value
    v = max(dictionary, key=dictionary.get) #Return the key for dictionary that has the max value (Centrality)
    return v

def minimum_distance(graph,start,goal):
    shortest = {} #Initialize an empty dictionary that will contain each visited node and its distance from the starting node
    shortest[start] = 0 #Put as a key the starting node which has zero distance
    visited = set([start]) #Create a set of nodes that i visted
    neighbour = set(graph.neighbors(start)) #Define the neighbors of start node
    non_visited = set() #I make a set of nodes that have not been visited.
    start_distance = 0 #I initialize the variable that indicates the starting distance equal to zero
    while not all(x in visited for x in goal): #I continue the process until all pages associated with the category are in the set of visited nodes
        visited = visited.union(neighbour) #I join the neighbors of the node I am in to the set of visited nodes as I will certainly visit them
        while len(neighbour) > 0: #Until the len of set neighbors is major of zero (Exist neighbors)
            current_vertex = neighbour.pop() #Inizializzo il vertice su cui mi trovo
            if (current_vertex in goal) and (current_vertex not in shortest.keys()): #Se tale vertice si trova nella mia lista di pagine da raggiungere e non è gia come chiave del dizionario
                shortest[current_vertex] = start_distance + 1  #Pongo questo nuovo vertice nel dizionario e associo distanza uguale a start_distance ottenuta fino a quel punto +1  
            for element in set(graph.neighbors(current_vertex)): #ricerco ogni vicino di current_vertex
                if element not in visited: #se quel nodo non è in quelli gia visitati
                    non_visited.add(element) #Initialize the vertex I am on
        
        if len(non_visited) == 0: #Se non ho nodi da visitare mi fermo
            break
        neighbour = non_visited #Unvisited nodes become the new neighbors to go to
        non_visited = set() #Clear unvisited nodes
        start_distance += 1 #I increase the distance made by one since the graph has weighted edges with a value of 1
       
    print("Partial number Click obtained before the node on which one has arrived has no more neighbors is:", start_distance)
    if nx.is_weakly_connected(graph) == False: #I ask if Graph is connected or not
        return print("Not Possible because Graph is not Connected")
    else:
        if (nodes - shortest.keys())!=0:#If not all nodes are keys of the shortest dictionary I have not reached all pages
            print("From the start nodes", v, " we can't reach all pages in nodes")
        else:
            print("From the start nodes", v, "reach all pages in nodes with number of click", start_distance) #I reach all pages in nodes list with a minimum distance equal to start_distance
        
            
def sub_graph (graph, c1, c2, df):
    
    c1_nodes = df[df['Category']== c1]['PageList']     #nodes of c1
    lst1 = [int(val) for sublist in c1_nodes for val in sublist] #list of nodes of c1
    
    c2_nodes = df[df['Category']== c2]['PageList']     #nodes of c2
    lst2 = [int(val) for sublist in c2_nodes for val in sublist] #list of nodes of c2
    
    nodes = lst1+lst2 #list of the nodes of c1 AND c2
    
    sub_graph = graph.subgraph(nodes).copy() #building the subgraph
    
    return nx.DiGraph(sub_graph)

def ford_fulkerson(graph, s, t):
    n = len(graph)
    INF = float('Inf')
    max_flow = 0
 
    f = [[0 for k in range(n)] for i in range(n)] #flow matrix
 
    while True:  # while there is a path from s to t 
        # Use BFS to find s-t path 
        prev = [ -1 for _ in range(n) ] # This array is filled by BFS and to store path
        prev[s] = -2 # s marked as visited
        q = deque()
        q.append(s) # s is visited
 
        while q and prev[t] == -1: #while q is not empty and t is not visited
            u = q.popleft() #visit u
            for v in range(n):
                path_flow = graph[u][v] - f[u][v]  # path_flow = graph(u,v) - f(u,v) for the edge (u,v)
                if path_flow > 0 and prev[v] == -1: #if is possible to have a path and v is not visited
                    q.append(v) #visit v
                    prev[v] = u #store u in index v
 
        if prev[t] == -1:   # if t has not been visited
            break
 
        # augment s-t path in residual graph that has been found by BFS
        v = t
        delta = INF
        while True:
            u = prev[v]
            path_flow = graph[u][v] - f[u][v]
            delta = min(delta, path_flow)
            v = u
            if v == s:
                break
 
        v = t
        while True:
            u = prev[v]
            f[u][v] += delta #increase the flow
            f[v][u] -= delta #decrease the flow
            v = u
            if v == s:
                break
 
        max_flow += delta
 
    return max_flow #min_cut


def shortpath(graph, source, articles):

    v = [source]
    lvl = [source]
    
    distances = dict.fromkeys(articles, -1)
    count = 1
    while True:
        new_lvl = []
        for article in lvl:
            for adj in graph.incident_edges(article):
                if adj not in v:
                    new_lvl.append(adj)
                    v.append(adj)
                    if adj in articles:
                        distances[adj] = count
        if len(new_lvl) == 0:
            break
        lvl = new_lvl
        count +=1
    
    if (np.array(list(distances.values())) == -1).all():
        return -1
    else:
        shortest_distance = min([distance for article, distance in distances.items() if distance != -1])
        
    return shortest_distance

def c_distance(graph, categories, input_category):

    cate_distances = {}
    
    for category, articles in categories.items():
        shortest_path = {}
        articles = [int(article) for article in articles if article in graph.vertices()]
        if category != input_category:
            for source in categories[input_category]:
                if source in graph.vertices():
                    shortest_distance = shortpath(graph, int(source), articles)
                    
                    if shortest_distance == -1: 
                         continue
                    else:
                        shortest_path[source] = shortest_distance
            
            if len(shortest_path) == 0:
                continue
            else:
                median = np.median([shortest for source, shortest in shortest_path.items()])
                cate_distances[category] = median
    if len(cate_distances) == 0:
        return 'Input Category have no connection with the other categories'
    else:
        return dict(sorted(cate_distances.items(), key=lambda item: item[1]))
def PageRank(G, d = 0.85, max_iter = 100):
    probs_node = { x: 1 / len(G) for x in G.nodes }                                                                   
    out_degrees = { x: G.out_degree(x) if G.out_degree(x) != 0 else len(G) for x in G.nodes }                         
    
    new_probs_node = probs_node.copy()                                                                                 
                                                                                                                       
    for _ in tqdm(range(max_iter)):                                                                                    
        for p_x in new_probs_node:                                                                                                         
            ratios_sum = sum([ probs_node[neigh] / out_degrees[neigh] for neigh in nx.neighbors(G, p_x) ])             
            new_probs_node[p_x] = ( ( 1 - d ) / len(G) ) + ( d * ratios_sum )                                          
        
            probs_node = new_probs_node.copy()
    
    return new_probs_node
