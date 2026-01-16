import numpy as np
import random
import os
import argparse

from agents import Agent
from util import Language, normalize_probs, normalize_logprobs, gibbs_sampler, build_hierarchical_graph, create_edge, pick_based_on_age, select_parents
from math import log, exp, log2
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import networkx as nx

import math

def type_of_language(language):
    # hard coded for small meaning space
    words = []
    for meaning, signal in language:
        words.append(signal) 
    if len(set(words)) == 1:
        return 'degenerate'
    if len(set(words)) == 4 and ((words[0][0] == words[1][0] and words[2][0] == words[3][0] and words[0][1] == words[2][1] and words[1][1] == words[3][1]) or
     (words[0][1] == words[1][1] and words[2][1] == words[3][1] and words[0][0] == words[2][0] and words[1][0] == words[3][0])):
        return 'compositional'
    if len(set(words)) == 4:
        return 'holistic'
    return 'other'


def generate_grammar(language):
    """ 
    function to generate minimally redundant forms of languages
    hardcoded for the small signal meaning space.     
    """

    holistic_rules = defaultdict(list) # signal to (list of) meanings

    A1 = defaultdict(set); B1 = defaultdict(set)  # direct: m1->s1, m2->s2
    A2 = defaultdict(set); B2 = defaultdict(set)  # swapped: m1->s2, m2->s1

    for meaning, signal in language:
        holistic_rules[signal].append(meaning)
        if len(meaning) == 2 and len(signal) == 2:
            m1, m2 = meaning[0], meaning[1]
            s1, s2 = signal[0], signal[1]

            # orientation 1 (direct)
            A1[m1].add(s1)
            B1[m2].add(s2)

            # orientation 2 (swapped)
            A2[m1].add(s2)
            B2[m2].add(s1)
    

    # degenerate (single signal)
    if len(holistic_rules) == 1:
        return ".".join(f"S{','.join(holistic_rules[word])}{word}" for word in holistic_rules)

    def is_compositional(A, B):
        return all(len(v) == 1 for v in A.values()) and all(len(v) == 1 for v in B.values())

    # prefer direct orientation if both true
    if is_compositional(A1, B1):
        A_rules = ".".join(f"A{k}{''.join(v)}" for k, v in sorted(A1.items()))
        B_rules = ".".join(f"B{k}{''.join(v)}" for k, v in sorted(B1.items()))
        header = "SAB"
        return f"{header}.{A_rules}.{B_rules}"

    if is_compositional(A2, B2):
        A_rules = ".".join(f"A{k}{''.join(v)}" for k, v in sorted(A2.items()))
        B_rules = ".".join(f"B{k}{''.join(v)}" for k, v in sorted(B2.items()))
        header = "SBA"
        return f"{header}.{A_rules}.{B_rules}"

    # fallback: holistic listing
    return ".".join(f"S{','.join(sorted(holistic_rules[word]))}{word}" for word in holistic_rules)


def code_length(grammar: str) -> float:
    N = len(grammar)
    freqs = Counter(grammar)
    return sum(-math.log2(freqs[ch] / N) for ch in grammar)


def log_non_normed_prior(grammar: str, beta: float = 1.0):
    L_bits = code_length(grammar)
    return beta * -L_bits * math.log(2)

def update_posterior(posterior, meaning, signal, signals, error_probability, languages):
    
    log_prior = np.asarray(posterior)
    
    # Likelihoods
    in_language = log(1 - error_probability)
    out_of_language = log(error_probability / (len(signals) - 1))
    
    language_matrix = np.array([((meaning, signal) in lang) for lang in languages])
    likelihood = np.where(language_matrix, in_language, out_of_language)
    
    # Update posterior
    new_log_posterior = log_prior + likelihood
    new_log_posterior = normalize_logprobs(new_log_posterior)

    return new_log_posterior


def sample(posterior, languages, mode):
    if mode == 'map':
        return languages[np.argmax(posterior)]  # Take most probable language
    elif mode == 'sample':
        exp_post = np.exp(posterior - np.max(posterior))  # Stability trick
        probabilities = exp_post / np.sum(exp_post)
        return languages[np.random.choice(len(languages), p=probabilities)]
    else:
        raise ValueError("Invalid mode. Choose either 'MAP' or 'sample'.")

def literal_listener(language, signal, meanings):
    """
    Takes language (list of four 2-tuples), signal (two-character string, e.g., 'aa'), and all possible meanings.
    Returns a meaning that the literal listener associates with this signal.
    """
    possibles = []
    for m, s in language:
        if s == signal:
            possibles.append(m) # Possibles ends up with all the meanings that are mapped to the signal
    if possibles == []:
        return np.random.choice(meanings) # If we don't have any meanings for the signal, just guess!
    else:
        return np.random.choice(possibles) # Otherwise, pick one of the possible meanings

def literal_speaker(language, meaning, signals, error_probability):
    """
    Takes language (list of four 2-tuples), meaning (two-digit string, e.g., '02'), all possible signals, 
    and error probability.
    Returns a signal that the speaker produces for this meaning.
    """
    for m, s in language:
        if m == meaning:
            signal = s # find the signal that is mapped to the meaning 
                       # (nb. there's no synonymy possible in this model!)
  
    if random.random() < error_probability: # add the occasional mistake
        other_signals = []
        for other_signal in signals:
            if other_signal != signal:
                other_signals.append(other_signal) # make a list of all the "wrong" signals
        return np.random.choice(other_signals) # pick one of them
    
    return signal

def pragmatic_speaker(language, meaning, meanings, signals, error_probability):
    """
    Takes language (list of four 2-tuples), meaning (two-digit string, e.g., '02'), all possible meanings, 
    all possible signals, and error probability.
    Returns a signal that the pragmatic speaker produces for this meaning.
    """
    signal = literal_speaker(language, meaning, signals, error_probability)
    listener_meaning = literal_listener(language, signal, meanings) # check what a listener would think that signal would mean
    if listener_meaning != meaning:
        signal = np.random.choice(signals) # if the intended meaning is different from the received one, 
                                        # pick a different signal at random 
    return signal


def population_communication(population, population_dict, languages, meanings, signals, 
                             error_probability, mode, round_with_one_agent, 
                             compress_prior):
   
    signaller_index = random.randrange(0, len(population)) # select random signaller from population

    signaller = population[signaller_index]

    learner_id = np.random.choice(signaller.neighbors) # select a random neighbor from signaller
    learner = population_dict[learner_id]
    for _ in range(round_with_one_agent):
        meaning = np.random.choice(meanings)
        s_language = sample(signaller.prior, languages, mode)
    
        signal = pragmatic_speaker(s_language, meaning, meanings, signals, error_probability) # pragmatic signal

        if random.random() < error_probability:
            other_signals = [s for s in signals if s != signal]
            signal = random.choice(other_signals)

        learner.prior = update_posterior(learner.prior, meaning, signal, signals, error_probability, languages)


def language_stats(population, generation, run, turnover_rounds, group_rounds, popsize, types, languages):
    stats = {'run': run, 'generation': generation, 'degenerate': 0, 'holistic': 0, 'other': 0, 'compositional': 0, 
             'popsize' : popsize, 'turnover_rounds': turnover_rounds, 'group_rounds': group_rounds}
    for agent in population:
        posteriors = np.exp(agent.prior)
        for i, p in enumerate(posteriors):
            stats[types[i]] += p
 

    k = ['degenerate', 'holistic', 'compositional', 'other']
    for key in stats.keys():
        if key in k:
            stats[key] = stats[key] / len(population)

    return stats


def generate_network(network_type, pop_size):

    if network_type == "fully-connected":
        G = nx.complete_graph(pop_size)
        return G
    
    elif network_type == "small-world":
        G = nx.connected_watts_strogatz_graph(n = pop_size, k = 4, p = 0.1)
        return G
    
    elif network_type == "scale-free":
        G1 = nx.scale_free_graph(pop_size)
        G = nx.Graph(G1)
        G.remove_edges_from(nx.selfloop_edges(G))
        while not nx.is_connected(G):
            G1 = nx.scale_free_graph(pop_size)
            G = nx.Graph(G1) 
            G.remove_edges_from(nx.selfloop_edges(G))
        return G
    
    elif network_type == "hierarchical":
        # cheching if population size is 5^x (i.e. 5, 25, 125, 625...)
        n = round(log(pop_size, (5)), 9)
        if not n.is_integer(): 
            raise Exception("For the hierarchical network the number of agents must be 5^x, where x >= 1.")
        else:
            n = int(n - 1)
            final_graph = build_hierarchical_graph(n)
        return final_graph

    elif network_type == "mobile-phone":
        # hard coded parameter values to get network with specific clustering coefficient and average degree
        iterations = 3000
        beta1= 500
        beta2= 10000
        initial_graph = nx.erdos_renyi_graph(pop_size, 0.25)
        beta = 1.677
        target_mean_degree = pop_size ** (beta - 1)
        target_clustering = 0.25
        final_graph = gibbs_sampler(initial_graph, iterations, beta1, beta2, target_mean_degree, target_clustering)
        return final_graph
    
    else:   
        raise Exception("Network type should be fully-connected, hierarchical, small-world, scale-free, or mobile-phone.")


def pick_based_on_communicative_success(population, languages, mode, meanings, signals, error_probability):
    """ function that has every agent communicate with every other agent 
        agent is picked based low communicative success."""
    
    com_suc_pop = []
    for signaller in population:
        com_suc_agent = 0
        for receiver in population:
            if signaller != receiver:
                meaning = np.random.choice(meanings)

                s_language = sample(signaller.prior, languages, mode)
                signal = literal_speaker(s_language, meaning, signals, error_probability)

                l_language = sample(receiver.prior, languages, mode)
                guessed_meaning = literal_listener(l_language, signal, meanings)

                if meaning == guessed_meaning:
                    com_suc_agent += 1
            
        com_suc_pop.append(com_suc_agent)
    
    # change into probs
    com_suc_probs = normalize_probs(com_suc_pop)

    com_suc_probs_reverse = []
    
    m = max(com_suc_probs)
    for prob in com_suc_probs:
        com_suc_probs_reverse.append(m - prob)

    com_suc_probs_reverse = normalize_probs(com_suc_probs_reverse)

    chosen_agent = np.random.choice(population, p=com_suc_probs_reverse)

    return chosen_agent

                


def simulation(generations, languages, meanings, signals, rounds, popsize, turnover_rounds, 
               run, error_probability, mode, compress_prior, replace, types,
               network_type, dynamic, edge_add, edge_remove, init_rounds):
    results = []
    population_dict = {}

    
    # picking a random holistic language for the first generation
    holistic = []
    for i, lang in enumerate(types):
        if lang == 'holistic':
            holistic.append(languages[i])

    initial_lang_idx = np.random.choice(list(range(len(holistic))))
    initial_lang = holistic[initial_lang_idx]

    initial_lang_dict = {}

    for meaning, form in initial_lang:
        initial_lang_dict[meaning] = form

    print(initial_lang_dict)


    np.random.seed(run)
    random.seed(run)

    mode = mode.lower()

    network = generate_network(network_type, popsize)

    # saving some properties of the network
    network_data = {}
    network_data['type'] = network_type
    network_data['average_degree'] = np.mean([deg for _, deg in network.degree()])
    network_data['clustering_coef'] = nx.average_clustering(network)
    network_data['average_shortest_path'] = nx.average_shortest_path_length(network)
    network_data['n_edges'] = len(list(network.edges()))
    network_data['n_nodes'] = len(list(network.nodes()))

    data = pd.DataFrame(network_data, index=[0])

    # initialize population
    population = [Agent(id=i, prior=compress_prior.copy(), age=0, neighbors = [n for n in network.neighbors(i)]) for i in range(popsize)]

    # create a dict to keep track of id to agent
    for agent in population:
        population_dict[agent.id] = agent

    new_id = len(population) + 1
    extra_agent_to_add = 0

    # intial learning rounds for entire population
    for agent in population:
        for r in range(init_rounds):
            meaning = np.random.choice(meanings)
            signal = initial_lang_dict[meaning]
            agent.prior = update_posterior(agent.prior, meaning, signal, signals, error_probability, languages)

    np.save(f"prior_{init_rounds}", np.array(population[0].prior))

    r = language_stats(population, 0, run, turnover_rounds, rounds, popsize, types, languages)
    results.append(r)
    
    print(f"h_trans{rounds}, v_trans{turnover_rounds}, pop_size{popsize}, mode{mode}, replace{replace}")
    for generation in tqdm(range(generations)):
        # horizontal transmission
        for r in range(rounds):
            population_communication(population, population_dict, languages, meanings, signals, error_probability, 
                                     mode, rounds, compress_prior)
        r = language_stats(population, generation + 1, run, turnover_rounds, rounds, popsize, types, languages)
        results.append(r)
        #print(r)

        if dynamic:
            # increment age of current population 
            for agent in population:
                agent.age += 1

            # new agent is added in a random spot, connected to two parents
            if turnover_rounds > 0:
                for i in range(extra_agent_to_add + 1): # add at least one new agent, and additionally extra if more were removed previously
                    parent1, parent2 = select_parents(population, population_dict)
                    parents = [parent1, parent2]
                    new_agent = Agent(id=new_id, prior=compress_prior.copy(), neighbors = [parent1.id, parent2.id], age=0)
                    population.append(new_agent)
                    population_dict[new_id] = new_agent
                    
                    # actually add them to the network
                    network.add_node(new_id)
                    network.add_edges_from([(new_id, parent1.id), (new_id, parent2.id)])

                    # increment for the next new agent
                    new_id += 1

                    # let new agent learn from one of the parents:
                    for _ in range(turnover_rounds):
                        signaller = np.random.choice(parents)
                        signaller_posterior = signaller.prior

                        learner_posterior = new_agent.prior
                            
                        meaning = np.random.choice(meanings)

                        s_language = sample(signaller_posterior, languages, mode)
                        signal = pragmatic_speaker(s_language, meaning, meanings, signals, error_probability)

                        new_agent.prior = update_posterior(learner_posterior, meaning, signal, signals, error_probability, languages)
            
            # to keep track of changes to the network
            edges_to_create = []
            edges_to_remove = []

           
            # for each agent, remove or add edges with probability p
            for agent in population:
                agent.neighbors = [n for n in network.neighbors(agent.id)]

                p_add = random.random()
                if edge_add < p_add:
                    connect_to = create_edge(agent, 0, population_dict, population)
                    if connect_to is not None and connect_to != agent.id and not network.has_edge(agent.id, connect_to):
                        edges_to_create.append((agent.id, connect_to))
                p_remove = random.random()
                if edge_remove < p_remove and agent.neighbors:
                    remove = np.random.choice(agent.neighbors)
                    if network.has_edge(agent.id, remove):
                        edges_to_remove.append((agent.id, remove))

            network.add_edges_from(edges_to_create)
            network.remove_edges_from(edges_to_remove)
        
            # one agent dies per generation (for now based on age only)
            pick_based_on_communicative_success(population, languages, mode, meanings, signals, error_probability)
            old_agent, old_agent_id = pick_based_on_age(population)
            del population_dict[old_agent.id]
            population.remove(old_agent)
            network.remove_node(old_agent.id)

            # update neighbors lists and let them age
            extra_agent_to_add = 0
            agents_to_remove = []
            for agent in population:
                agent.neighbors = [n for n in network.neighbors(agent.id)]
                # prune agent from network if they do not have any neighbors
                if agent.neighbors == []:
                    agents_to_remove.append(agent.id)
                    extra_agent_to_add += 1
                agent.age += 1
            
            for agent_id in agents_to_remove:
                agent = population_dict[agent_id]
                del population_dict[agent_id]
                network.remove_node(agent_id)
                population.remove(agent)
    
        else:
            # replacing agents in the population based on age, new agent takes the place of the old agent
            for agent in population:
                agent.age += 1
            if turnover_rounds > 0:
                for _ in range(replace):    
                    old_population = population.copy()   
                    picked_agent = np.random.choice(population) 
                    #picked_agent, to_be_removed_id = pick_based_on_age(population)
                
                    new_agent = Agent(id=picked_agent.id, prior=compress_prior.copy(), neighbors = [n for n in network.neighbors(picked_agent.id)], age=0)
                
                    for _ in range(turnover_rounds):
                            
                        meaning = np.random.choice(meanings)

                        s_language = sample(picked_agent.prior, languages, mode)
                        signal = pragmatic_speaker(s_language, meaning, meanings, signals, error_probability) # pragmatic signal
                            
                        new_agent.prior = update_posterior(new_agent.prior, meaning, signal, signals, error_probability, languages)
                    old_population.remove(picked_agent)
                    population = [new_agent] + old_population

    return pd.DataFrame(results), data

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got '{v}'")

def main():
    parser = argparse.ArgumentParser(description="Run a single simulation instance")
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--group_round", type=int, required=True)
    parser.add_argument("--turnover_round", type=int, required=True)
    parser.add_argument("--pop_size", type=int, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--network_type", type=str, required=True)
    parser.add_argument("--dynamic", type=str2bool, required=True)
    parser.add_argument("--replace", type=str2bool, required=True)
    parser.add_argument("--beta", type=float, required=True)

    args = parser.parse_args()

    generations = 10000
    error_probability = 0.05

    SIGNAL_SPACE = 'ab'
    language = Language(SIGNAL_SPACE)
    meanings = language.return_meanings()
    signals = language.return_signals()
    languages = language.return_languages()
    types = [type_of_language(lang) for lang in languages]

    gr_strings = [generate_grammar(lang) for lang in languages]
    priors = [log_non_normed_prior(g, args.beta) for g in gr_strings]
    priors = normalize_logprobs(priors)

    
    # if True replace entire population, else replace one agent
    r = args.pop_size if args.replace else 1

    edge_add = 0.3
    edge_remove = 0.3


    result, data = simulation(
        generations, languages, meanings, signals,
        args.group_round, args.pop_size, args.turnover_round,
        args.run_id, error_probability, args.mode,
        priors, r, types, args.network_type, args.dynamic, edge_add, 
        edge_remove, 0
    )

    results_dir = f"results_snellius/{args.mode}/{args.replace}/{args.beta}"
    
    os.makedirs(results_dir, exist_ok=True)
    result.to_csv(f"{results_dir}/run{args.run_id}.csv", index=False)
    # os.makedirs(f"{results_dir}/network_data", exist_ok=True)
    # data.to_csv(f"{results_dir}/network_data/run{args.run_id}.csv", index=[0])

if __name__ == "__main__":

    main()
