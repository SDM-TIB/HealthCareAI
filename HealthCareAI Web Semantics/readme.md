# The steps to reproducing results in our paper

1. (Do not need to run) the dataset is quried from the original knowledge graph (KG) using the jupyter notebook `data/query_data.ipynb`. We remove the orignial dataset: `data/1808_original.csv` since we do not have the right to make it publicly avaiable. 
   
2. (Run the code) The jupyter notebook `casual/casual_discovery.ipynb` is used to simulate KGs (with different settings of patient number N, which is the `sample_size` in the code) and counterfactuals from dataset queried from the the original KG, and learn causal graph from the simulated KGs. 
The Horn rules are mined using `rule_mining/rule_mining.ipynb`. Rules with `PCA confident = 1` are used as a part of the domain knowledge. 
The LLM prompts are presented in `casual/casual_discovery.ipynb`.
The final learned causal graphs for each N (`sample_size`) are stored in `causal/structures_N.pkl`. 

3. (Run the code) The python file `causal_reasoning.py` is used to esitmate counterfactuals from the simulated KGs. Specify the parameter `sample_size` for each patient number N. 
