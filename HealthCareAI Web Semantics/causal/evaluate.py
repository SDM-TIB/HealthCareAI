import pandas as pd
import random
from bn_library import pc_causal_discovery, fci_causal_discovery, ges_causal_discovery, K2
from sklearn.model_selection import GroupKFold

# Assuming the necessary import statements for the causal discovery functions are included.

def format_latex_table(df):
    # Convert the DataFrame to a LaTeX table
    return df.to_latex(index=False, float_format="%.3f", column_format="lccc")


def run_method_multiple_times(method_function, df, times=15):
    results = []
    for _ in range(times):
        structure = method_function(df, causal_direction=True)
        results.append(structure)
    return results


def evaluate_causal_structures(structures, gold_standard):
    results = []
    optimized_structures = {method: set() for method in structures}

    for method, all_predictions in structures.items():
        method_results = []
        for predictions in all_predictions:
            predictions_set = set(predictions)
            intersection = gold_standard.intersection(predictions_set)
            union = gold_standard.union(predictions_set)

            jaccard_index = len(intersection) / len(union) if union else 0
            precision = len(intersection) / len(predictions_set) 
            recall = len(intersection) / len(gold_standard)
            F1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0

            method_results.append({
                'Jaccard Index': jaccard_index,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': F1
            })

            # Keep track of the best precision structures
            if precision > max((re['Precision'] for re in method_results), default=0):
                optimized_structures[method] = predictions_set

        avg_results = pd.DataFrame(method_results).mean().to_dict()
        avg_results['Method'] = method
        results.append(avg_results)

    results_df = pd.DataFrame(results)
    results_df = results_df[['Method', 'Precision', 'Recall', 'F1-Score', 'Jaccard Index']]
    return results_df, optimized_structures

def structure_learn_evalutation(structure_df, times = 15):
    random.seed(123)
    # structure_df = pd.DataFrame()  # Replace with actual DataFrame loading or creation

    methods = {
        'pc': pc_causal_discovery,
        'fci': fci_causal_discovery,
        'ges': ges_causal_discovery,
        'K2': K2,
    }

    # Run each method 15 times
    structures = {method: run_method_multiple_times(func, structure_df, times) for method, func in methods.items()}
    structures['gpt4'] = [[
    ("Age", "Biomarker", '->'),  # Causal
    ("SmokerType", "Biomarker", '->'),  # Causal
    ("Gender", "SmokerType", '->'),  # Causal
    ("Gender", "Biomarker", '-'),  # Correlated
    ("FamilyCancer", "Biomarker", '-'),  # Correlated
    ("FamilyDegree", "FamilyCancer", '-')  # Correlated
    ]*times]

    # Expert structure as gold standard
    gold_standard = set([('SmokerType', 'Biomarker', '->'), 
    ('Age', 'SmokerType', '->'), ('Age', 'Biomarker', '->'), 
    ('Gender', 'SmokerType', '->'), ('Gender', 'Biomarker', '->'),
    ('FamilyGender', 'Biomarker', '->'), ('FamilyGender', 'FamilyCancer', '->'),
    ('FamilyDegree', 'Biomarker', '->'),
    ('FamilyCancer', 'Biomarker', '->')])  # Assume the expert's structure is consistently provided

    # Evaluate causal dependencies
    causal_results, optimized_causal_structures = evaluate_causal_structures(structures, gold_standard)
    # print(format_latex_table(causal_results))

    # Evaluate dependencies without direction consideration
    undirected_structures = {
        key: [{(e[0], e[1]) for e in val} | {(e[1], e[0]) for e in val} for val in structures[key]]
        for key in structures
    }
    dependency_results, optimized_dependency_structures = evaluate_causal_structures(undirected_structures, gold_standard)
    # print(format_latex_table(dependency_results))
    return optimized_causal_structures, causal_results, dependency_results

from sklearn.model_selection import StratifiedKFold

def create_stratified_folds(df, Y='Biomarker', T='SmokerType', random_seed = 123):
    # Seed for reproducibility
    
    # Creating a new column that represents the combination of X and Y
    df['TY'] = df[T].astype(str) + df[Y].astype(str)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    
    # Dictionary to hold the indices for each fold
    folds = {
        'fold_{}'.format(i+1): {
            'train_set': None,
            'test_set': None
        } for i in range(5)
    }

    # Generate the folds
    for i, (train_index, test_index) in enumerate(skf.split(df, df['TY'])):
        folds['fold_{}'.format(i+1)]['train_set'] = train_index
        folds['fold_{}'.format(i+1)]['test_set'] = test_index

    # Cleanup the temporary column
    df.drop(columns=['TY'], inplace=True)
    
    return [val for val in folds.values()]

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_categorical_distribution(df):
    """
    Visualize the distribution of each categorical column in a DataFrame using bar charts,
    with both subplots and bins ordered alphabetically.

    Args:
    df (pandas.DataFrame): DataFrame with categorical columns to visualize.
    """
    col_size = 4
    # Set up the aesthetics for the plots
    sns.set(style="whitegrid")
    
    # Sort dataframe columns alphabetically
    sorted_columns = sorted(df.columns)
    
    # Calculate the number of rows needed for subplots based on the number of categorical columns
    num_columns = len(sorted_columns)
    num_rows = (num_columns + 1) // col_size  # Adjust this line to change how many columns per row you prefer
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=col_size, figsize=(14, 4 * num_rows))  # Adjust figure size as needed
    axes = axes.flatten()  # Flatten the array of axes if they are in matrix form
    
    # Loop through the sorted columns and create a countplot for each
    for i, column in enumerate(sorted_columns):
        # Order categories alphabetically within the column for the plot
        ordered_data = df[column].dropna().astype(str).sort_values().unique()
        sns.countplot(x=df[column], ax=axes[i], order=ordered_data, palette='viridis')
        axes[i].set_title(f'Distribution of {column}', fontsize=14)
        axes[i].set_xlabel(column, fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")  # Rotate labels for better fit
    
    # If the number of columns in the dataframe is odd, delete the last subplot (which is empty)
    if num_columns % col_size != 0:
        fig.delaxes(axes[-1])
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(edge_list, title):
    """
    Draw a graph based on a list of tuples (A, B, l) where:
    - A, B are node labels
    - l is a string defining the type of edge ('->', '-', '<->')
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges based on the edge type
    for A, B, l in edge_list:
        if l == '->':
            G.add_edge(A, B)
        elif l == '-':
            G.add_edge(A, B)
            G.add_edge(B, A)
        elif l == '<->':
            G.add_edge(A, B)
            G.add_edge(B, A)
    
    # Set up the plot with matplotlib
    plt.figure(figsize=(8, 5))

    pos = nx.spring_layout(G)  # Positions for all nodes
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='k', linewidths=1, font_size=15, arrows=True)
    
    # Show the graph
    plt.title(title)
    plt.show()

# Example of usage
# edges = [
#     ('A', 'B', '->'),
#     ('B', 'C', '-'),
#     ('C', 'D', '<->')
# ]

# draw_graph(edges)



'''
import pandas as pd

def evaluate_causal_structures(structures):
    # Extract the gold standard set of tuples
    gold_standard = set(structures['expert'])
    
    # Initialize a list to hold results
    results = []
    
    # Iterate through each method and compute metrics
    for method, predictions in structures.items():
        if method == 'expert':
            continue
        predictions = set(predictions)
        # Intersection and union for Jaccard Index
        intersection = gold_standard.intersection(predictions)
        union = gold_standard.union(predictions)
        jaccard_index = len(intersection) / len(union) if union else 0
        
        # Precision and Recall calculations
        precision = len(intersection) / len(predictions) if predictions else 0
        recall = len(intersection) / len(gold_standard) if gold_standard else 0
        F1 = 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

        # Formatting results to percentage with 1 decimal place
        # print(jaccard_index, precision, round(precision, 3), recall, F1)
        jaccard_percent = round(jaccard_index, 3)
        precision_percent = round(precision, 3)
        recall_percent = round(recall, 3)
        F1 = round(F1, 3)
        
        # Append results to the list
        results.append({
            'Method': method,
            'Jaccard Index': jaccard_percent,
            'Precision': precision_percent,
            'Recall': recall_percent,
            'F1-Score': F1
        })
    
    # Create DataFrame from the results list
    results_df = pd.DataFrame(results)
    
    # Return the dataframe
    return results_df

def format_latex_table(df):
    # Convert the DataFrame to a LaTeX table
    return df.to_latex(index=False, float_format="%.3f", column_format="lccc")

result_table = evaluate_causal_structures(structures)
print(format_latex_table(result_table))

structures = {key: {(e[0], e[1]) for e in val} | {(e[1], e[0]) for e in val} for key, val in structures.items()}
result_table = evaluate_causal_structures(structures)
print(format_latex_table(result_table))

'''