import numpy as np
import pandas as pd

'''[('SmokerType', 'Biomarker'), 
 ('Age', 'SmokerType'), ('Age', 'Biomarker'), 
 ('Gender', 'SmokerType'), ('Gender', 'Biomarker'),
 ('FamilyGender', 'Biomarker'),
 ('FamilyDegree', 'Biomarker'),
 ('FamilyCancer', 'Biomarker')]
'''

def generate_data(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Define variable mappings
    variable_mappings = {
        'Age': {0: 'Young', 1: 'Old'},
        'Gender': {0: 'Male', 1: 'Female'},
        'FamilyGender': {0: 'Woman', 1: 'Men', 2: 'WomanAndMen'},
        'FamilyDegree': {0: 'FirstDegree', 1: 'SecondDegree', 2: 'ThirdDegree'},
        'FamilyCancer': {0: 'Major', 1: 'Minor', 2: 'MajorAndMinor'},
        'SmokerType': {0: 'Smoker', 1: 'Non-Smoker'},
        'Biomarker': {1: 'ALKorEGFR', 0: 'Others'},
        'Biomarker1': {1: 'ALKorEGFR', 0: 'Others'},
        'Biomarker0': {1: 'ALKorEGFR', 0: 'Others'}
    }
    
    data = pd.DataFrame()
    
    # Generate exogenous variables
    data['Age'] = np.random.randint(0, 2, size=N)  # 0 or 1
    data['Gender'] = np.random.randint(0, 2, size=N)
    data['FamilyGender'] = np.random.randint(0, 3, size=N)  # 0,1,2
    data['FamilyDegree'] = np.random.randint(0, 3, size=N)  # 0,1,2
    data['FamilyCancer'] = np.random.randint(0, 3, size=N)  # 0,1,2
    
    # Generate SmokerType based on Age and Gender
    data['SmokerType'] = generate_SmokerType(data['Age'], data['Gender'])
    
    # Generate Biomarker (potential outcomes and observed outcome)
    (data['Biomarker'], data['Biomarker0'], data['Biomarker1'],
     data['p_Y0'], data['p_Y1'], data['p_Effect']) = generate_Biomarker(
        data['SmokerType'], data['Age'], data['Gender'], data['FamilyGender'],
        data['FamilyDegree'], data['FamilyCancer'])
    
    # Map numerical data to categorical data
    categorical_data = map_to_categorical(data, variable_mappings)
    
    return data, categorical_data

def generate_SmokerType(Age, Gender):
    N = len(Age)
    # Define coefficients
    b0 = 0.0
    b1 = 0.5  # Coefficient for Age
    b2 = -0.5  # Coefficient for Gender
    noise = np.random.normal(0, 0.1, N)
    
    # Compute linear function
    lin = b0 + b1*Age + b2*Gender + noise
    # Compute probability using sigmoid
    prob_smoker = 1 / (1 + np.exp(-lin))
    
    # Sample from Bernoulli distribution
    SmokerType = np.random.binomial(1, prob_smoker)
    
    return SmokerType  # 0: 'Smoker', 1: 'Non-Smoker'

def generate_Biomarker(SmokerType, Age, Gender, FamilyGender, FamilyDegree, FamilyCancer):
    N = len(SmokerType)
    # Define coefficients
    b0 = 0.0
    b_treat = 0.9  # Treatment effect (SmokerType)
    b_age = -0.3    # Coefficient for Age
    b_gender = 0.3  # Coefficient for Gender
    b_familygender = 0.2  # Coefficient for FamilyGender
    b_familydegree = -0.1  # Coefficient for FamilyDegree
    b_familycancer = -0.1  # Coefficient for FamilyCancer
    noise = np.random.normal(0, 0.1, N)
    
    # Potential outcome when SmokerType=0 (Smoker)
    lin0 = (b0 + b_treat*0 + b_age*Age + b_gender*Gender + 
            b_familygender*FamilyGender + b_familydegree*FamilyDegree +
            b_familycancer*FamilyCancer + noise)
    prob_Y0 = 1 / (1 + np.exp(-lin0))
    Biomarker0 = np.random.binomial(1, prob_Y0)
    
    # Potential outcome when SmokerType=1 (Non-Smoker)
    lin1 = (b0 + b_treat*1 + b_age*Age + b_gender*Gender + 
            b_familygender*FamilyGender + b_familydegree*FamilyDegree +
            b_familycancer*FamilyCancer + noise)
    prob_Y1 = 1 / (1 + np.exp(-lin1))
    Biomarker1 = np.random.binomial(1, prob_Y1)
    
    # Observed outcome depends on actual treatment
    Biomarker = np.where(SmokerType == 0, Biomarker0, Biomarker1)
    
    # Treatment effect
    p_Effect = prob_Y1 - prob_Y0
    
    return Biomarker, Biomarker0, Biomarker1, prob_Y0, prob_Y1, p_Effect

def map_to_categorical(data, variable_mappings):
    categorical_data = data.copy()
    for column in data.columns:
        mapping = variable_mappings.get(column)
        if mapping:
            categorical_data[column] = data[column].map(mapping)
    return categorical_data



# # Display first few rows of numerical data
# print("Numerical Data:")
# print(numerical_data.head())

# # Display first few rows of categorical data
# print("\nCategorical Data:")
# print(categorical_data.head())
