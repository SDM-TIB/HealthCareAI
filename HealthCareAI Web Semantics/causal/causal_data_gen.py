import pandas as pd
import numpy as np
import statsmodels.api as sm


# values in in baseline_dict corresponds to 0
baseline_dict = {"Age": "Young", "Gender": "Female", "SmokerType": "Smoker", 'Biomarker': 'Others', 'FamilyCancer':"hasMinor",
       'FamilyGender': "Woman", 'FamilyDegree': "Degree1"} #,   'FamilyDiversity': "1"}
reverse_smoker_map = {1: 'Non-Smoker', 0: "Smoker"}             # basline_dict indicates that "Smoker" is 0
reverse_family_cancer_map = {0: 'hasMinor', 1: 'OnlyMajor'}     # basline_dict indicates that "hasMinor" is 0
reverse_biomarker_map = {1: 'ALKorEGFR', 0: 'Others'}           # basline_dict indicates that "Others" is 0

# Logistic Regression Model Inputs
features_M1 = ['Age', 'Gender']
target_M1 = 'SmokerType'
features_M2 = ['FamilyGender']
target_M2 = 'FamilyCancer'
features_M3 = sorted(['Age', 'Gender', 'SmokerType', 'FamilyGender', 'FamilyCancer', 'FamilyDegree']) #, 'FamilyDiversity'])
target_M3 = 'Biomarker'

sigma = 0.1

variable_order = sorted([baseline_dict.keys()])


def create_dummies(df, baseline):
    df = df[sorted([e for e in baseline.keys()])]
    dummies = pd.get_dummies(df, drop_first=False)
    
    # Remove baseline dummies specified in baseline
    for col, baseline_val in baseline.items():
        baseline_dummy = f"{col}_{baseline_val}"
        if baseline_dummy in dummies.columns:
            dummies.drop(columns=[baseline_dummy], inplace=True)
    dummies = dummies.astype(int)
    return dummies


def train_model(df, y_col, y_value, baseline, show_p=False):
    df = df.copy()
    df[y_col] = (df[y_col] == y_value).astype(int)
    
    df = df[sorted([e for e in baseline.keys()] + [y_col])]
    dummies = pd.get_dummies(df.drop(columns=[y_col]), drop_first=False)
    
    # Remove baseline dummies specified in baseline
    for col, baseline_val in baseline.items():
        baseline_dummy = f"{col}_{baseline_val}"
        if baseline_dummy in dummies.columns:
            dummies.drop(columns=[baseline_dummy], inplace=True)
    dummies = dummies.astype(int)
        
    X = sm.add_constant(dummies)
    y = df[y_col]
    
    model = sm.Logit(y, X).fit(disp=0)
    if show_p:
        print(pd.DataFrame({"Coefficients": model.params,"Odds":np.exp(model.params), "P-Value": model.pvalues}))
    else:
        print(pd.DataFrame({"Coefficients": model.params,"Odds":np.exp(model.params)}))
    return model


# Step 1: Load data
def step1(data: pd.DataFrame):
    
    # Model M1
    model_M1 = train_model(df=data, y_col=target_M1, y_value='Non-Smoker', baseline={key: baseline_dict[key] for key in features_M1})
    # model_M1 = {target_M1+"_"+smoker_type: train_model(data, y_col=target_M1, y_value=smoker_type, baseline_dict={key: baseline_dict[key] for key in features_M1}) for smoker_type in data[target_M1].unique()}

    # Model M2
    model_M2 = train_model(df=data, y_col=target_M2, y_value='OnlyMajor', baseline={key: baseline_dict[key] for key in features_M2})

    # Model M3
    model_M3 = train_model(data, y_col=target_M3, y_value='ALKorEGFR', baseline={key: baseline_dict[key] for key in features_M3})
    
    # from descriptive.odds_ratio import calculate_odds_ratios
    # calculate_odds_ratios(df=data, y_col='Biomarker', y_value='ALKorEGFR', baseline_dict={"Age": "Young", "Gender": "Female", "SmokerType": "Current-Smoker", 'FamilyCancer':"hasMinor",
    #    'FamilyGender': "Woman", 'FamilyDegree': "Degree1", 'FamilyDiversity': "1"})
    return model_M1, model_M2, model_M3


def step2(data: pd.DataFrame):
    # Step 2: Calculate the distribution probabilities of certain categorical variables
    def calculate_distribution(data, columns):
        distributions = {}
        for column in columns:
            # Count the frequency of each category
            count = data[column].value_counts(normalize=True)
            distributions[column] = count
        return distributions

    # Define the columns for which to calculate distributions 
    columns = ['Age', 'Gender', 'FamilyGender', 'FamilyDegree'] #, 'FamilyDiversity']

    # Calculate distributions
    distributions = calculate_distribution(data, columns)

    # Display the distributions
    # for column, distribution in distributions.items():
    #     print(f"Distribution for {column}:\n{distribution}\n")
    return distributions


def step3(distributions, num_samples=2000, uniform=True):
    
    # Step 3: Generate new categorical values based on the calculated distributions
    def generate_categorical_values(distributions, num_samples):
        generated_data = {}
        for column, distribution in distributions.items():
            categories = distribution.index
            probabilities = distribution.values
            if uniform:
                probabilities = np.ones_like(probabilities) / probabilities.size
            # Generate random samples based on the distribution probabilities
            generated_samples = np.random.choice(categories, size=num_samples, p=probabilities)
            generated_data[column] = generated_samples
        return pd.DataFrame(generated_data)

    # Generate 5000 categorical values for the specified columns
    generated_data = generate_categorical_values(distributions, num_samples)

    # print("-------------------------------step2-------------------------------")
    # step2(generated_data)

    # Display the first few rows of the generated data
    # print(generated_data.head())
    return generated_data


def step4(generated_data, model_M1, model_M2, model_M3):
    # Step 4: Generate values using the additive noisy model

    # Function to calculate probability using logistic regression coefficients and added noise
    def calculate_probability(X, coefficients, noise_std=sigma):
        z = np.dot(X, coefficients)                                 # Calculate logit (z) from coefficients and features
        z += np.random.normal(0, noise_std, size=X.shape[0])        # Add Gaussian noise
        probability = 1 / (1 + np.exp(-z))                          # Convert logit to probability using sigmoid function
        return probability

    # def m1_probability(X_M1, model_M1:dict):
    #     ps = []
    #     vals = []
    #     for smoker_type, model_m1 in model_M1.items():
    #         vals.append(smoker_type[smoker_type.index("_")+1:])
    #         ps.append(calculate_probability(X_M1, model_m1.params))
    #     ps = np.vstack(ps).T 
    #     ps = ps / ps.sum(axis=1, keepdims=True)
    #     return vals, ps
    
    def generate_values(generated_data, model_M1, model_M2, model_M3):
        # Prepare features for each model, the order of columns i X_M1, X_M2, X_M3 are matter 
        X_M1 = sm.add_constant(create_dummies(df=generated_data, baseline={key: baseline_dict[key] for key in features_M1}))
        X_M2 = sm.add_constant(create_dummies(df=generated_data, baseline={key: baseline_dict[key] for key in features_M2}))

        # Calculate probabilities
        # val1, p1 = m1_probability(X_M1, model_M1)
        # generated_data['SmokerType'] = [np.random.choice(val1, p=p) for p in p1]
        p1 = calculate_probability(X_M1, model_M1.params)
        p2 = calculate_probability(X_M2, model_M2.params)
        
        # Simulate outcomes based on probabilities
        generated_data['SmokerType'] = [reverse_smoker_map[i] for i in np.random.binomial(1, p1)] # [reverse_smoker_map[1 if p > 0.5 else 0] for p in p1]# 
        generated_data['FamilyCancer'] = [reverse_family_cancer_map[i] for i in np.random.binomial(1, p2)] # [reverse_family_cancer_map[1 if p > 0.5 else 0] for p in p2]   # 

        # Generate biomarker under interventions
        X_M3 = sm.add_constant(create_dummies(df=generated_data, baseline={key: baseline_dict[key] for key in features_M3})) #, 'FamilyDiversity'])))
        
        X_M3['SmokerType_Non-Smoker'] = 1  # Intervention do(SmokerType='Non-Smoker')
        p_Y1 = calculate_probability(X_M3, model_M3.params)
        X_M3['SmokerType_Non-Smoker'] = 0  # Intervention do(SmokerType='Smoker')
        p_Y0 = calculate_probability(X_M3, model_M3.params)
        
        generated_data['Biomarker0'] = [reverse_biomarker_map[1 if p > 0.5 else 0] for p in p_Y0] # [reverse_biomarker_map[i] for i in np.random.binomial(1, p_Y0)]
        generated_data['Biomarker1'] = [reverse_biomarker_map[1 if p > 0.5 else 0] for p in p_Y1] #[reverse_biomarker_map[i] for i in np.random.binomial(1, p_Y1)]
        # generated_data['Biomarker0'] = [reverse_biomarker_map[i] for i in np.random.binomial(1, p_Y0)]
        # generated_data['Biomarker1'] = [reverse_biomarker_map[i] for i in np.random.binomial(1, p_Y1)]
        
        generated_data['Biomarker'] = np.where(generated_data['SmokerType'] == 'Non-Smoker', generated_data['Biomarker1'], generated_data['Biomarker0'])
        generated_data['p_Y1'] = p_Y1
        generated_data['p_Y0'] = p_Y0
        generated_data['p_Effect'] = p_Y1 - p_Y0
        return generated_data
    
    # Generate the new values based on the additive noisy model
    generated_data_with_outcomes = generate_values(generated_data, model_M1, model_M2, model_M3)

    # Display the first few rows of the data with generated outcomes
    # print(generated_data_with_outcomes.head())
    return generated_data_with_outcomes

import pandas as pd
def generate_tsv(df, output_file):
    """
    Generate a TSV file with triples from a DataFrame, where each row represents a patient,
    and triples are generated for each patient's properties.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing patient data.
        output_file (str): Path to the output TSV file.
        
    Returns:
        None
    """
    # Open a file to write the triples
    with open(output_file, 'w') as file:
        # Write header
        # file.write("PatientID\tProperty\tValue\n")
        
        # Iterate over each row in the DataFrame
        for idx, row in df.iterrows():
            # Create a unique patient ID for each row
            patient_id = f"Patient_{idx + 1}"
            
            # Generate triples for each property
            for col in df.columns:
                property_name = col
                value = row[col]
                
                # Write the triple to the TSV file
                file.write(f"{patient_id}\t{property_name}\t{value}\n")


