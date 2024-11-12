import pandas as pd
from evaluate import create_stratified_folds
import sys , os
import pickle
from pgmpy.models import BayesianNetwork

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sample_size = 10000

generated_data_with_outcomes = pd.read_csv(f"data/{sample_size}_gen.csv")
structure_df = generated_data_with_outcomes.drop(columns=['Biomarker1', 'Biomarker0', 'p_Y1', 'p_Y0', 'p_Effect'])

causal_graphs = None
with open(f"causal/structures_{sample_size}.pkl", 'rb') as file:
    causal_graphs = pickle.load(file)

from evaluate import create_stratified_folds
from bn_library import inference, do_model
import pandas as pd
import numpy as np
from bn_library import MLE
from scipy.stats import pearsonr

for k, g in causal_graphs.items():
    print(k, ":", g)


def evaluate_one_method(bn: BayesianNetwork, df:pd.DataFrame, Y=['Biomarker'], T1={'SmokerType': "Non-Smoker"}, T0={'SmokerType': "Smoker"}):
    # P(Y|X=x, do(SmokerType='Non-Smoker')), PP(Y|X=x, do(SmokerType='Smoker'))
    evidence_cols = [c for c in df.columns if c not in Y and c not in T1.keys()]
    p_Y1 = []
    p_Y0 = []
    y_CF = []
    
    # obtaining twin networks
    M1 = do_model(bn, interventions=T1)
    M0 = do_model(bn, interventions=T0)
    T = list(T1.keys())[0]

    print(bn.edges, bn.nodes, df.columns)
    df = df[[col for col in df.columns if col in bn.nodes and col not in Y]]
    for i, row in df.iterrows():
        # row = {k:v for k, v in row.items() if k in bn.nodes}
        res1 = inference(M1, query_nodes=Y, evidences={**{k:v for k, v in row.items() if k != T}, **T1}) 
        res0 = inference(M0, query_nodes=Y, evidences={**{k:v for k, v in row.items() if k != T}, **T0})
        p_Y1.append(res1.get_value(Biomarker="ALKorEGFR"))
        p_Y0.append(res0.get_value(Biomarker="ALKorEGFR"))

        if row[T] == T0[T]:
            y_CF.append("ALKorEGFR" if res1.get_value(Biomarker="ALKorEGFR") > 0.5 else "Others")
        if row[T] == T1[T]:
            y_CF.append("ALKorEGFR" if res0.get_value(Biomarker="ALKorEGFR") > 0.5 else "Others")

    p_Y1 = np.array(p_Y1)
    p_Y0 = np.array(p_Y0)
    
    return p_Y1, p_Y0, y_CF

correlations = {key:[] for key in causal_graphs.keys()}

for fold in create_stratified_folds(generated_data_with_outcomes):
    training, testing = generated_data_with_outcomes.iloc[fold['train_set']], generated_data_with_outcomes.iloc[fold['test_set']]
    drop_columns = 'Biomarker0,Biomarker1,p_Y1,p_Y0,p_Effect'.split(',')
    causal_nets = {key: MLE(dag, training.drop(columns=drop_columns)) for key, dag in causal_graphs.items()}
    for method, bn in causal_nets.items():
        # print(method, bn)
        p_Y1, p_Y0, y_CF = evaluate_one_method(bn, testing.drop(columns=drop_columns+['Biomarker']))
        y_CF_ground = np.where(testing['SmokerType'].to_numpy() == 'Non-Smoker', testing['Biomarker0'], testing['Biomarker1'])
        
        to_num = {"ALKorEGFR":1, "Others":0}
        correlation, _ = pearsonr([to_num[e] for e in y_CF], [to_num[e] for e in y_CF_ground])
        correlations[method].append(correlation)

# Calculate the average of the absolute differences and correlations across all folds
avg_correlation = {key: (np.mean(val), np.std(val)) for key, val in correlations.items()}

with open('avg_correlation.pkl', 'wb') as f:
    pickle.dump(avg_correlation, f)


def print_res(res):
    for key, value in res.items():
        print(f"{key}: ({value[0]:.4f}, ±{value[1]:.4f})")

print("")
print(f"Average Pearson Correlation:\n", )
print_res(avg_correlation)

