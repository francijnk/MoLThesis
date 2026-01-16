import random
import networkx as nx
import numpy as np
from itertools import product
from scipy.special import logsumexp
from math import log, exp, log2
from scipy.spatial.distance import jensenshannon

class Language:
    def __init__(self, signal_space):

        self.signal_space = signal_space

        self.length = len(self.signal_space)
        self.signals = [''.join(comb) for comb in product(signal_space, repeat=len(signal_space))]

        self.symbol_sets = [list(range(i * self.length, (i + 1) * self.length)) for i in range(self.length)]
        self.meanings = ["".join(map(str, combination)) for combination in product(*self.symbol_sets)]

    def return_meanings(self):
        return self.meanings
    
    def return_signals(self):
        return self.signals
    
    def return_languages(self):
        languages = [[(self.meanings[0], s0), (self.meanings[1], s1), (self.meanings[2], s2), (self.meanings[3], s3)] 
             for s0 in self.signals for s1 in self.signals for s2 in self.signals for s3 in self.signals]
        return languages


########### General Helper Functions ##################
   
def normalize_probs(probs):
    total = sum(probs) #calculates the summed probabilities
    normedprobs = []
    for p in probs:
        normedprobs.append(p / total) 
    return normedprobs

def normalize_logprobs(logprobs):
    logtotal = logsumexp(logprobs) #calculates the summed log probabilities
    normedlogs = []
    for logp in logprobs:
        normedlogs.append(logp - logtotal) #normalise - subtracting in the log domain equivalent to divising in the normal domain
    return normedlogs


def log_roulette_wheel(normedlogs):
    """Takes a list of normed log probabilities; returns some index of that list 
    with probability corresponding to the (exponentiated) value of that list element"""
    r=log(random.random()) #generate a random number in [0,1), then convert to log
    accumulator = normedlogs[0]
    for i in range(len(normedlogs)):
        if r < accumulator:
            return i
        accumulator = logsumexp([accumulator, normedlogs[i + 1]])

################### Hierarchical network ###################

def build_hierarchical_graph(levels):

    # Base case fully connected graph
    G = nx.complete_graph(5)
    total_nodes = 5

    # Track which nodes in G corner nodes 
    # first nodes 1–4 are the corner nodes of G
    peripheral_nodes = [1, 2, 3, 4]

    for _ in range(1, levels + 1):
        prev = G.copy()
        newG = prev.copy()
        new_peripherals = []

        # Create 4 replicas of `prev`
        for _ in range(4):
            # Relabel nodes by offset to avoid collisions
            mapping = {n: n + total_nodes for n in prev.nodes()}
            replica = nx.relabel_nodes(prev, mapping, copy=True)

            # Merge replica into newG
            newG.add_nodes_from(replica.nodes(data=True))
            newG.add_edges_from(replica.edges(data=True))

            # Collect just the top-level corner nodes of this replica
            new_peripherals.extend(mapping[n] for n in peripheral_nodes)

            total_nodes += prev.number_of_nodes()

        # Connect each of those new corner nodes to the **original** center (node 0)
        for v in new_peripherals:
            newG.add_edge(0, v)

        # Prepare for next iteration
        G = newG
        peripheral_nodes = new_peripherals

    return G


################### Functions for mobile phone graph - based on code provided by Reali et al. (2018):
# https://github.com/mhchristiansen/lang-paradox/blob/master/RandomNetworksFuns2.R  ################

def hamiltonian(G, beta1, beta2, target_mean_degree, target_clustering):
    mean_deg = np.mean([deg for _, deg in G.degree()])
    clust = nx.average_clustering(G)
    h = beta1 * (mean_deg - target_mean_degree) ** 2 + beta2 * (clust - target_clustering) ** 2
    return h, mean_deg, clust



# === GIBBS SAMPLER ===
def gibbs_sampler(G, iterations, beta1, beta2, target_mean_degree, target_clustering):
    prev_h, prev_deg, prev_clust = hamiltonian(G, beta1, beta2, target_mean_degree, target_clustering)

    nodes = list(G.nodes())

    for _ in range(iterations):
        # select two random nodes
        u, v = random.sample(nodes, 2)
        if u == v:
            continue

        # add or remove edge
        if G.has_edge(u, v):
            G.remove_edge(u, v)
        else:
            G.add_edge(u, v)

        # check hamiltonian, clustering and degreee of new network
        h_new, deg_new, clust_new = hamiltonian(G, beta1, beta2, target_mean_degree, target_clustering)
        accept_prob = np.exp(prev_h - h_new)

        if random.random() < accept_prob:
            # Accept
            prev_h, prev_deg, prev_clust = h_new, deg_new, clust_new
        else:
            # Reject: revert change
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            else:
                G.add_edge(u, v)

    return G


########## Dynamic Network Helper Functions ###############

def select_parents(population, population_dict):
    n_neighbors = 0
    while n_neighbors == 0:
        parent1idx = np.random.randint(len(population))
        parent1 = population[parent1idx]
        parent2 = parent1
        n_neighbors = len(parent1.neighbors)
    counter = 0

    # making sure they are not the same agent
    while parent2 == parent1:
        counter += 1
        parent2id = np.random.choice(parent1.neighbors)
        parent2 = population_dict[parent2id]
        if counter > 1000:
            raise Exception("stuck here")
    return parent1, parent2

def pick_based_on_age(population):
    agent_list = []
    for agent in population:
        l = [agent.id] * agent.age
        agent_list.extend(l)

    chosen_agent_id = np.random.choice(agent_list)
    for agent in population:
        if agent.id == chosen_agent_id:
            picked_agent = agent
    return picked_agent, chosen_agent_id



def pick_k_similar_agents(agent, k, population):
    a_list = []
    distances = {}
    for a in population:
        result = jensenshannon(agent.prior, a.prior)
        distances[a.id] = result

    sorted_list = sorted(distances.items(), key=lambda x: x[1])

    for agent in range(k):
        a_list.append(sorted_list[agent][0])
    
    return a_list

def create_edge(agent, ratio, population_dict, population, network):
    """ 
        agent will create a new edge
        with probability ratio the edge will be created based on transitivity,
        and 1 - ratio it is based on homophilly.
        
        """
    
    for agent in population:
        agent.neighbors = [n for n in network.neighbors(agent.id) if n in population_dict]
    
    p = random.random()
    if p > ratio:
    # transitivity - put neighbors of neighbors in a list
        transitivity_options = []
        for neighbor in agent.neighbors:
            n = population_dict[neighbor]
            transitivity_options.extend(n.neighbors)
        connect_to_agent = np.random.choice(transitivity_options)
        return connect_to_agent
    else:
        # pick 10 agents who are most similar to this agent
        options = pick_k_similar_agents(agent, 10, population)
        return np.random.choice(options)
