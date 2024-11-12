# import bnlearn as bn
# from matplotlib.pyplot import axis
# from importlib_metadata import Prepared
# from matplotlib.pyplot import axis
from pgmpy.estimators import K2Score, BicScore, BDeuScore
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, PC, TreeSearch, MmhcEstimator
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.inference import VariableElimination, CausalInference
from pgmpy.factors.discrete import TabularCPD
from copy import deepcopy
import random
import pyAgrum as gum
import itertools
import numpy as np
from causallearn.search.FCMBased import lingam
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from sklearn.preprocessing import LabelEncoder



################################################
# Structure Learning
# bnlearn: Hillclimbsearch, Chow-liu, TAN, Constraint-based
# reference: https://ermongroup.github.io/cs228-notes/learning/structure/
# pgmpy: PC (constraint-based), Hill Climb Search, Tree Search, Mmhc, Exhaustive search
# reference: https://pgmpy.org/structure_estimator/base.html
################################################

score_funs = {'bic': BicScore, 'k2': K2Score, 'bdeu': BDeuScore}

def encode_data(df):
    # This function encodes categorical data to numeric using LabelEncoder
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

def pc_causal_discovery(df, causal_direction=False):
    df_encoded = encode_data(df)
    # Apply FCI algorithm
    cg = pc(df_encoded.values)
    A = cg.G.graph 
    causal_pairs = []
    for i in range(A.shape[0]):
        for j in range(i+1, A.shape[1]):
            if A[i,j] == -1 and A[j,i] == 1:
                if causal_direction:
                    causal_pairs.append((df.columns[i], df.columns[j], "->"))
                else:
                    causal_pairs.append((df.columns[i], df.columns[j]))
            elif A[i,j] == -1 and A[j,i] == -1:
                if causal_direction:
                    causal_pairs.append((df.columns[i], df.columns[j], "-"))
                else:
                    causal_pairs.append((df.columns[i], df.columns[j]))
            elif A[i,j] == 1 and A[j,i] == 1:
                if causal_direction:
                    causal_pairs.append((df.columns[i], df.columns[j], "<->"))
                else:
                    causal_pairs.append((df.columns[i], df.columns[j]))
    return causal_pairs


def fci_causal_discovery(df, causal_direction=False):
    # Encode data
    df_encoded = encode_data(df)
    # Apply FCI algorithm
    g, edges = fci(df_encoded.values)
    A = g.graph
    causal_pairs = []
    for i in range(A.shape[0]):
        for j in range(i+1, A.shape[1]):
            if A[i,j] == -1 and A[j,i] == 1:
                if causal_direction:
                    causal_pairs.append((df.columns[i], df.columns[j], "->"))
                else:
                    causal_pairs.append((df.columns[i], df.columns[j]))
            elif A[i,j] == 2 and A[j,i] == 1:
                if causal_direction:
                    causal_pairs.append((df.columns[i], df.columns[j], "->"))
                else:
                    causal_pairs.append((df.columns[i], df.columns[j]))
            elif A[i,j] == 2 and A[j,i] == 2:
                if causal_direction:
                    causal_pairs.append((df.columns[i], df.columns[j], "-"))
                else:
                    causal_pairs.append((df.columns[i], df.columns[j]))
            elif A[i,j] == 1 and A[j,i] == 1:
                if causal_direction:
                    causal_pairs.append((df.columns[i], df.columns[j], "<->"))
                else:
                    causal_pairs.append((df.columns[i], df.columns[j]))
    return causal_pairs

def ges_causal_discovery(df, causal_direction=False):
    # Encode data
    df_encoded = encode_data(df)
    # Apply GES algorithm
    Record = ges(df_encoded.values)
    g = Record['G']
    
    A = g.graph
    causal_pairs = []
    for i in range(A.shape[0]):
        for j in range(i+1, A.shape[1]):
            if A[i,j] == -1 and A[j,i] == 1:
                if causal_direction:
                    causal_pairs.append((df.columns[i], df.columns[j], "->"))
                else:
                    causal_pairs.append((df.columns[i], df.columns[j]))
            elif A[i,j] == -1 and A[j,i] == -1:
                if causal_direction:
                    causal_pairs.append((df.columns[i], df.columns[j], "-"))
                else:
                    causal_pairs.append((df.columns[i], df.columns[j]))
    return causal_pairs



def lingam_method(df, causal_direction=False):
    for column in df.columns:
        df[column] = pd.Categorical(df[column]).codes
    model = lingam.DirectLiNGAM()
    model.fit(df.to_numpy())
    adjacency_mat = model.adjacency_matrix_
    causal_pairs = [(df.columns[i], df.columns[j], '->') for i in range(adjacency_mat.shape[0]) 
                        for j in range(adjacency_mat.shape[1]) if adjacency_mat[i][j] != 0] if causal_direction else [(df.columns[i], df.columns[j]) for i in range(adjacency_mat.shape[0]) 
                        for j in range(adjacency_mat.shape[1]) if adjacency_mat[i][j] != 0]
    return causal_pairs


# 1. score based
def exhaustive_search(data, score_option='bic'):
    es = ExhaustiveSearch(data, scoring_method=score_funs[score_option](data))
    best_model = es.estimate()
    return list(best_model.edges())
    # print(best_model.edges())

def hill_climate(data, score_option='bic', causal_direction=False):
    # based on simulated annealing
    model = HillClimbSearch(data).estimate(scoring_method=score_funs[score_option](data))
    return [(e[0], e[1], '->')for e in model.edges()] if causal_direction else list(model.edges())
    # assert score_option in score_funs.keys()
    # model = bn.structure_learning.fit(data, methodtype='hc', scoretype='bic')
    # model = bn.independence_test(model, data, alpha=0.05, prune=True)
    # # 
    # return model['model_edges'] #.edges()

def TAN(data, root_node=None, score_option='bic'):
    '''Tree-augmented Naive Bayes, a tree based structure
    '''
    tan = TreeSearch(data, root_node=root_node)
    model = tan.estimate(estimator_type='chow-liu')
    return list(model.edges())
    # assert score_option in score_funs.keys()
    # model = bn.structure_learning.fit(data, methodtype='tan', class_node=root_node, scoretype=score_option)
    # model = bn.independence_test(model, data, alpha=0.05, prune=True)   # use chi-squre test to remove no significant edges
    # return model['model_edges']

def K2(data, score_option='bic', causal_direction=False):
    # reference https://agrum.gitlab.io/articles/agrumpyagrum-0229-and-dataframe.html
    s_learner = gum.BNLearner(data)  # creates a learner by passing the dataframe
    # s_learner.useGreedyHillClimbing()     # sets a local-search algorithm for the structural learning
    s_learner.useK2(random.shuffle([i for i in range(len(data.columns))]))  # using random the typology order
    if score_option == 'bic':
        s_learner.useScoreBIC()              # sets BIC score as the metric
    elif score_option == 'k2':
        s_learner.useScoreK2()
    elif score_option == 'bdeu':
        s_learner.useScoreBDeu()
    else:
        raise Exception("wrong score option")
    structure_learn = s_learner.learnBN()       # learning the structure
    id2name = {structure_learn.idFromName(node_n): node_n for node_n in structure_learn.names()}
    return [(id2name[ele[0]], id2name[ele[1]], '->') for ele in structure_learn.arcs()] if causal_direction else [(id2name[ele[0]], id2name[ele[1]]) for ele in structure_learn.arcs()]

    # print(structure.arcs())
    # print(structure.nodes())
    # # print(train_data.columns)
    # print(structure.names())
    # # print(structure.cpt('Rain'))
    # print(structure.idFromName('Rain'))
    # structure.dag()

# def mcmc_edges(a):
    

# 2. constraint based
# def constraint_based(data):
#     c = PC(data)
#     model_chi = c.estimate(ci_test='chi_square', return_type='dag')
#     return list(model_chi.edges())
    

# 3. hybrid method
def mmhc(data, score_option='bic', causal_direction=False):
    assert score_option in score_funs.keys()
    mmhc = MmhcEstimator(data)
    model = mmhc.estimate()
    return [(e[0], e[1], '->')for e in model.edges()] if causal_direction else list(model.edges())

################################################
# Parameter Learning
################################################
# 1. Maximum Likelihood Estimator
def MLE(skeleton, data):
    from pgmpy.estimators import MaximumLikelihoodEstimator
    model = BayesianNetwork(skeleton)
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model

# 2. Bayesian Estimator
def BE(skeleton, data):
    from pgmpy.estimators import BayesianEstimator
    model = BayesianNetwork(skeleton)
    model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")
    
    return model


################################################
# Predition and Inference
################################################

def inference(model, query_nodes, evidences, interprete=True):
    '''model: Bayesian Network
    '''
    infer = VariableElimination(model)
    res = infer.query(variables=query_nodes, evidence=evidences, show_progress=False)

    if interprete:
        interpret_result(res, query_nodes)
    return res


################################################
# Intervention
################################################

# def do(model, interventions):
#     from pgmpy.factors.discrete.CPD import TabularCPD
#     """ 
#     Implement an ideal intervention for discrete variables. Modifies pgmpy's
#     `do` method so it is a `do`-operator, meaning a function that takes in a
#     model, modifies it with an ideal intervention, and returns a new model.
#     Note that this code would need to be modified to work for continuous
#     variables.
#     """
#     def _mod_kernel(kernel, int_val):
#         """
#         Modify a causal Markov kernel so all probability is on the state fixed
#         by the intervention.
#         """ 
#         var_name = kernel.variable
#         card = kernel.get_cardinality([var_name])[var_name]
#         states = [kernel.get_state_names(var_name, i) for i in range(card)]
#         non_int_states = set(states) - {int_val,}
#         unordered_prob_vals = [[1.0]] + [[0.0] for _ in range(card - 1)]
#         unordered_states = [int_val] + list(non_int_states)
#         # Reorder so it matches original
#         dict_ = dict(zip(unordered_states, unordered_prob_vals))
#         ordered_prob_values = [dict_[k] for k in states]
#         intervention_kernel = TabularCPD(
#             var_name, card, ordered_prob_values,
#             state_names = {var_name: states}
#         )
#         return intervention_kernel

#     kernels = {kern.variable: kern for kern in model.get_cpds()}
#     new_model = model.copy()
#     for var, int_val in interventions.items():
#         new_model = new_model.do(var)
#         new_kernel = _mod_kernel(kernels[var], int_val)
#         new_model.add_cpds(new_kernel)
#     return new_model


# def do_model(original_model:BayesianNetwork, interventions:dict):
#     # Copy the original model for twin models
#     M1 = deepcopy(original_model)
#     for intervention_variable, value in interventions.items():
#         cpds = M1.get_cpds(node=intervention_variable)
#         one_pos = cpds.state_names[intervention_variable].index(value)
#         states = cpds.state_names[intervention_variable]
        
#         new_cpds = None
#         if one_pos == 1:
#             new_cpds = TabularCPD(variable=intervention_variable, variable_card=2, values=[[0], [1]], state_names={intervention_variable: states})
#         elif one_pos == 0:
#             new_cpds = TabularCPD(variable=intervention_variable, variable_card=2, values=[[1], [0]], state_names={intervention_variable: states})
        
#         M1.do(nodes=[intervention_variable])
#         M1.add_cpds(new_cpds)
#     return M1

def do_model(original_model: BayesianNetwork, interventions: dict):
    # Copy the original model to preserve the original
    M1 = deepcopy(original_model)
    
    for intervention_variable, value in interventions.items():
        # Remove edges coming into the intervention variable
        M1.remove_edges_from([(u, intervention_variable) for u, v in M1.in_edges(intervention_variable)])
        
        # Create a new CPD for the intervention variable
        states = original_model.get_cpds(node=intervention_variable).state_names[intervention_variable]
        one_pos = states.index(value)
        new_values = [[0]]*len(states)
        new_values[one_pos] = [1]
        # print("------", new_values)
        new_cpds = TabularCPD(variable=intervention_variable, variable_card=len(states), 
                              values=new_values, state_names={intervention_variable: states})
        
        M1.add_cpds(new_cpds)
        
    return M1


def causal_inference(model, query_nodes, interventions, evidences={}, interprete=True):
    '''model: Bayesian Network
    refer to https://pgmpy.org/examples/Causal%20Inference.html
    '''
    infer = CausalInference(model)
    
    # adj_variables = set([])
    # for Y, X in itertools.product(query_nodes, interventions.keys()):
    #     adj_variables |= infer.get_all_backdoor_adjustment_sets(X, Y)
    # if not adj_variables:
    #     adj_variables = None

    res = infer.query(variables=query_nodes, do=interventions, evidences=evidences, adjustment_set=None, show_progress=False)
    if interprete:
        interpret_result(res, query_nodes)
    return res

    # reference on https://colab.research.google.com/drive/1k8eoFkHugorOrjiH57bMXF84bV3ojzOm?usp=sharing#scrollTo=ZNBa1oGIOpyP 
    # from pgmpy.inference import CausalInference 
    # return CausalInference(model)
    # modified_model = do(model, interventions)
    # all_evidences = {**interventions, **evidences}
    # infer = VariableElimination(modified_model)
    # return inference(infer, query_nodes, all_evidences)


def interpret_result(res, query_nodes):
    state_names = {var: res.state_names[var] for var in query_nodes}
    for index, prob in np.ndenumerate(res.values):
        # 'index' will be a tuple with indices corresponding to the states of each variable
        states = [state_names[var][i] for var, i in zip(query_nodes, index)]
        # state_descr = ", ".join(f"{var}={state}" for var, state in zip(query_nodes, states))
        # print(f"{state_descr}: {prob:.4f}")


################################################
# Criteria
################################################

def get_y_and_pred(model, Y, data):
    y = data[Y].values.tolist()
    data=data.drop(Y, axis=1)
    pred = model.predict(data)[Y].values.tolist()
    # infer = VariableElimination(model)

    # y = data[Y].values.tolist()
    # y_dict = {e: i for i, e in enumerate(set(y))}   # for digitalize value
    # assert len(y_dict.keys()) == 2

    # data = data.drop(Y, axis=1)
    # pred = [infer.map_query([Y], row.to_dict())for _, row in data.iterrows()]

    # y = [y_dict[e] for e in y]
    # pred = [y_dict[e[Y]] for e in pred]
    return y, pred

def log_likelihood(model, data):
    from pgmpy.metrics import log_likelihood_score
    return log_likelihood_score(model, data)

def auc(model, Y, data):
    from sklearn import metrics
    y, pred = get_y_and_pred(model, Y, data)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    return metrics.auc(fpr, tpr)

def accuracy(model, Y, data):
    from sklearn.metrics import accuracy_score
    y, pred = get_y_and_pred(model, Y, data)
    return accuracy_score(y, pred)

