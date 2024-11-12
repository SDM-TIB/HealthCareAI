from collections import defaultdict
from rdflib.plugins.sparql import prepareQuery
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import json
from sklearn.model_selection import GroupKFold
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
                
def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def totalPopulation(input_data,endpoint):
	query="""SELECT count(DISTINCT ?ehr1) as ?num \n
	    WHERE {
		?ehr a <http://research.tib.eu/clarify2020/vocab/LCPatient>. 
		?ehr <http://research.tib.eu/clarify2020/vocab/has_LC_SLCG_ID> ?ehr1 . }"""
	sparql = SPARQLWrapper(endpoint)
	sparql.setQuery(query)
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	return int(results["results"]["bindings"][0]["num"]["value"])

def query_generation(input_data,endpoint,test=False):
	where_clause = {"Biomarker":"OPTIONAL {?ehr1 <http://research.tib.eu/clarify2020/vocab/hasBio> ?Biomarker.}.",
					"Relapse":"OPTIONAL {?ehr1 <http://research.tib.eu/clarify2020/vocab/hasProgressionRelapse> ?Relapse.}.",
					"Stages":"OPTIONAL {?ehr1 <http://research.tib.eu/clarify2020/vocab/hasDiagnosis> ?o1 . \n ?o1 <http://research.tib.eu/clarify2020/vocab/hasDiagnosisStage> ?Stages.}.",
					"Tumor":"OPTIONAL {?ehr1 <http://research.tib.eu/clarify2020/vocab/hasTumorHistology> ?Tumor.}.",
					"FamilyDegree":"""?ehr1 <http://research.tib.eu/clarify2020/vocab/hasFamilyHistory> ?o .\n ?o <http://research.tib.eu/clarify2020/vocab/familyRelationDegree> ?familyType .\n ?o  <http://research.tib.eu/clarify2020/vocab/hasFamilyCancerType> ?CancerType . \n FILTER (?CancerType != <http://research.tib.eu/clarify2020/entity/UNK>)""",
					"FamilyRelationship":"""?ehr1 <http://research.tib.eu/clarify2020/vocab/hasFamilyHistory> ?o .\n ?o <http://research.tib.eu/clarify2020/vocab/familyType> ?familyType .\n ?o  <http://research.tib.eu/clarify2020/vocab/hasFamilyCancerType> ?CancerType . \n FILTER (?CancerType != <http://research.tib.eu/clarify2020/entity/UNK>)""",}
	select_clause = {"Biomarker":"?Biomarker",
					"Relapse":"?Relapse",
					"Stages":"?Stages",
					"Tumor":"?Tumor"}
	query_select_clause = "SELECT DISTINCT ?ehr1 " # ?familyType ?CancerType"
	query_where_clause="""WHERE {
		?ehr a <http://research.tib.eu/clarify2020/vocab/LCPatient>. 
		?ehr <http://research.tib.eu/clarify2020/vocab/has_LC_SLCG_ID> ?ehr1 . \n"""
	# print(input_data)
	# if "Age" in input_data["Input"]["IndependentVariables"]:
	# 	query_where_clause = query_where_clause + "?ehr1  <http://research.tib.eu/clarify2020/vocab/age> ?age.\n"
	# 	query_select_clause = query_select_clause + "  ?age "
	# if "Gender" in input_data["Input"]["IndependentVariables"]:
	# 	query_where_clause = query_where_clause + "?ehr1 <http://research.tib.eu/clarify2020/vocab/sex> ?gender. FILTER (regex(?gender,\"" + input_data["Input"]["IndependentVariables"]["Gender"] + "\"))\n"
	# if "SmokingHabits" in input_data["Input"]["IndependentVariables"]:
	# 	query_where_clause = query_where_clause + "?ehr1 <http://research.tib.eu/clarify2020/vocab/hasSmokingHabit> ?smoking. FILTER (regex(?smoking,\"" + input_data["Input"]["IndependentVariables"]["SmokingHabits"] + "\"))\n"
	# if "FamilyType" in input_data["Input"]["IndependentVariables"].keys() and "FamilyDegree" == input_data["Input"]["IndependentVariables"]["FamilyType"]:
	# 	query_where_clause = query_where_clause + where_clause["FamilyDegree"] + "\n"
	# if "FamilyType" in input_data["Input"]["IndependentVariables"].keys() and "FamilyRelationship" == input_data["Input"]["IndependentVariables"]["FamilyType"]:
	# 	query_where_clause = query_where_clause + where_clause["FamilyRelationship"] + "\n"

	# for variable in input_data["Input"]["DependentVariables"]:
	# 	if variable != "CancerType":
	# 		query_select_clause = query_select_clause + select_clause[variable] + " "
	# 		query_where_clause= query_where_clause + where_clause[variable] + " \n"

	if "Age" in input_data["Input"]["IndependentVariables"]:
		query_select_clause += " ?age "
		query_where_clause += "?ehr1  <http://research.tib.eu/clarify2020/vocab/age> ?age . FILTER (?age != <http://research.tib.eu/clarify2020/entity/UNK>)\n"
	if "Gender" in input_data["Input"]["IndependentVariables"]:
		query_select_clause += " ?gender "
		query_where_clause += "?ehr1 <http://research.tib.eu/clarify2020/vocab/sex> ?gender . FILTER (?gender != <http://research.tib.eu/clarify2020/entity/UNK>)\n"
	if "SmokingHabits" in input_data["Input"]["IndependentVariables"]:
		query_select_clause += " ?smoking " 
		query_where_clause += "?ehr1 <http://research.tib.eu/clarify2020/vocab/hasSmokingHabit> ?smoking . FILTER (?smoking != <http://research.tib.eu/clarify2020/entity/UNK>)\n"
	if "FamilyRelationship" in input_data["Input"]["IndependentVariables"]: 
		query_select_clause += " ?familyType ?CancerType "
		query_where_clause += "?ehr1 <http://research.tib.eu/clarify2020/vocab/hasFamilyHistory> ?o .\n ?o <http://research.tib.eu/clarify2020/vocab/familyType> ?familyType .\n ?o  <http://research.tib.eu/clarify2020/vocab/hasFamilyCancerType> ?CancerType . \n FILTER (?CancerType != <http://research.tib.eu/clarify2020/entity/UNK> ).\n"
	if "Biomarker" in input_data["Input"]["IndependentVariables"]:
		query_select_clause += " ?biomarker "
		query_where_clause += "?ehr1 <http://research.tib.eu/clarify2020/vocab/hasBio> ?biomarker . FILTER (?biomarker != <http://research.tib.eu/clarify2020/entity/UNK>)\n"

	query_where_clause = query_where_clause[:-1] + "}"
	sparql_query = query_select_clause + " " + query_where_clause + ("LIMIT 50" if test else "")
	# print(sparql_query)
	# print(prepareQuery(sparql_query))

	sparql = SPARQLWrapper(endpoint)
	sparql.setQuery(sparql_query)
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	return results["results"]["bindings"]


def get_json_data(json_ls:list, prefix:str):
    return [{key: vals['value'].replace(prefix, "") for key, vals in ele.items()} for ele in json_ls]

def unique_values(json_ls: list):
    def vals(json_ls, key):
        return set(ele[key] for ele in json_ls)
    for k in json_ls[0].keys():
        if k in {'ehr1', 'Patient_id'}:
            continue
        print(k,":", vals(json_ls, k))


def pre_process(records):
    major_cancers = {'Breast', 'Lung', 'Colorectal', 'Head_and_neck', 'Uterus/cervical', 
                     'Esophagogastric', 'Prostate'}
    # major_cancers = {'Lung', 'Esophagogastric'}
    # https://chat.openai.com/share/6015bda0-e792-4e10-a4a6-16d5faf59fbf
    
    cancer_mapping = { # Thoracic Cancers or not
    'Lung': 'Thoracic Cancer',
    'Esophagogastric': 'Thoracic Cancer',  # Related due to proximity and possible metastasis.
    'Head_and_neck': 'Non Thoracic Cancer',  # Although thoracic, not directly linked to lung functionality.
    'Liver': 'Non Thoracic Cancer',
    'Skin_no_melanoma': 'Non Thoracic Cancer',
    'Renal': 'Non Thoracic Cancer',
    'Colorrectal': 'Non Thoracic Cancer',
    'Germinal_tumors': 'Non Thoracic Cancer',
    'Uterus/cervical': 'Non Thoracic Cancer',
    'Pancreatic': 'Non Thoracic Cancer',
    'Melanoma': 'Non Thoracic Cancer',
    'Gall_bladder': 'Non Thoracic Cancer',
    'Ovarian': 'Non Thoracic Cancer',
    'Sarcoma': 'Non Thoracic Cancer',
    'Prostate': 'Non Thoracic Cancer',
    'Other': 'Non Thoracic Cancer',
    'Leukemia': 'Non Thoracic Cancer',
    'Lymphoma': 'Non Thoracic Cancer',
    'Bladder/urinary_tract': 'Non Thoracic Cancer',
    'Breast': 'Non Thoracic Cancer',
    'Unknown_origin_carcinoma': 'Non Thoracic Cancer',
    'Central_nervous_system': 'Non Thoracic Cancer'
    }

    family_mapping = {
        "UNK": "UNK", "Father": "M1", "Mother": "F1", "Brother": "M1", "Sister": "F1",
        "Daughter": "F1", "Son": "M1", "Uncle": "M2", "Nephew": "M2", "Grandfather": "M2",
        "Grandmother": "F2", "Aunt": "F2", "Niece": "F2", "Granddaughter": "F2", "Grandson": "M2",
        "Grandgrandfather": "M3", "Grandgrandmother": "F3", "No": "No", "Halfsister": "F2",
        "Halfbrother": "M2", "Female_Cousin": "F3", "Male_Cousin": "M3", "NULL": "NULL", 
        'Granduncle': "M4", 'Greatgrandfather': "M4", 'Greatgrandmother':"F4"
    }
    
    processed = []
    
    for record in records:
        transformed = {}
        
        if 'ehr1' in record:
            transformed['Patient_id'] = record['ehr1']
        if 'age' in record:
            transformed['Age'] = "Young" if int(record['age']) <= 50 else "Old"
        if 'gender' in record:
            transformed['Gender'] = record['gender']
        if 'smoking' in record:
            transformed['SmokerType'] = ("Current-Smoker" if record['smoking'] == "CurrentSmoker" else
                        "Former-Smoker" if record['smoking'] == "PreviousSmoker" else
                        "Never-Smoker")
        if 'familyType' in record:
            transformed['Family'] = family_mapping[record['familyType']]
        if 'CancerType' in record:
            transformed['FamilyCancer'] = "Major" if record['CancerType'] in major_cancers else "Minor"
        if 'biomarker' in record:
            transformed['Biomarker'] = "ALKorEGFR" if record['biomarker'] in {"EGFR", "ALK"} else "Others"
        
        processed.append(transformed)
        
    return processed


def per_patient_data(processed_data):
    d_dict = {
            'Age': 0,
            'Gender': 0,
            'SmokerType': 1,
            'FamilyCancer': 1,
            'Biomarker': 1,
            'Family': 1
        }
    
    patient_data = {}

    # s1 = set([ele['Patient_id'] for ele in processed_data if ele['Biomarker'] == 'ALKorEGFR'])
    # s2 = set([ele['Patient_id'] for ele in processed_data if ele['Biomarker'] == 'Others'])
    # print(len(s1&s2), len(s1-s2), len(s2-s1)), print(list(s2-s1)[:4])

    # Aggregate data for each patient
    for record in processed_data:
        pid = record["Patient_id"]
        if not pid:
            continue  # Skip the record if 'Patient_id' is missing
        if pid in patient_data:
            for k in d_dict.keys():
                if k not in record:
                    continue
                if d_dict[k]:
                    patient_data[pid][k].add(record[k])  
                else:
                    patient_data[pid][k] = record[k]  
        else:
            patient_data[pid] = {}
            for k in d_dict.keys():
                if k not in record:
                    continue
                if d_dict[k]:
                    patient_data[pid][k] = set([record[k]])  # Initialize a new set for each field
                else:
                    patient_data[pid][k] = record[k]
    # print("----", patient_data['1214018'])
    # print([ele for ele in processed_data if ele['Patient_id'] == '1214018'])

    # Filter and transform data for each patient
    smoker_type_priority = ['Current-Smoker', 'Former-Smoker', 'Never-Smoker']
    biomarker_priority = ['ALKorEGFR', 'Others']

    result = []
    for pid, data in patient_data.items():
        if "Family" in data.keys():
            if 'UNK' in data['Family'] or 'NULL' in data['Family'] or 'No' in data['Family']:
                continue
                
        transformed = {"Patient_id": pid}
        for key in d_dict.keys():
            if key not in data:
                continue
            if key == 'Age':
                transformed[key] = data['Age']
            if key == "Gender":
                transformed[key] = data['Gender']
            if key == "SmokerType":
                transformed[key] = next((s for s in smoker_type_priority if s in data['SmokerType']), 'Never-Smoker')
            if key == "Biomarker":
                transformed[key] = next((b for b in biomarker_priority if b in data['Biomarker']), 'Others')
            if key == "FamilyCancer":
                transformed[key] = 'Major' if all(f.startswith('Major') for f in data['FamilyCancer']) else 'Minor' if all(f.startswith('Minor') for f in data['FamilyCancer']) else 'MajorandMinor'
            if key == "Family":
                transformed['FamilyGender'] = 'Woman' if all(f.startswith('F') for f in data['Family']) else 'Man' if all(f.startswith('M') for f in data['Family']) else 'ManorWoman'
                transformed['FamilyDegree'] = 'Degree1' if all(f.endswith('1') for f in data['Family']) else 'Degree2' if all(f.endswith('2') for f in data['Family']) else 'Degree3'
                transformed['FamilyDiversity'] = min(3, len(data['Family']))
        
        result.append(transformed)

    return result


def digitalize(aggregated_data):
    # Mapping definitions
    age_map = {'Young': -1, 'Old': 1}
    gender_map = {'Female': -1, 'Male': 1}
    smoker_type_map = {'Never-Smoker': 0, 'Former-Smoker': 1, 'Current-Smoker': 2}
    family_cancer_map = {'Minor': -1, 'Major': 1, 'MajorandMinor': 0}
    biomarker_map = {'Others': 0, 'ALKorEGFR': 1}
    family_degree_map = {'Degree1': 1, 'Degree2': 2, 'Degree3': 3}
    family_gender_map = {'Woman': -1, 'Man': 1, 'ManorWoman': 0}

    # Process each patient record to numerize
    result = []
    for record in aggregated_data:
        numerized = {'Patient_id': record['Patient_id']}  # Assuming 'Patient_id' is always present
        # Check if keys exist in the record and add to numerized dictionary
        if 'Age' in record:
            numerized['Age'] = age_map[record['Age']]
        if 'Gender' in record:
            numerized['Gender'] = gender_map[record['Gender']]
        if 'SmokerType' in record:
            numerized['SmokerType'] = smoker_type_map[record['SmokerType']]
        if 'FamilyCancer' in record:
            numerized['FamilyCancer'] = family_cancer_map[record['FamilyCancer']]
        if 'Biomarker' in record:
            numerized['Biomarker'] = biomarker_map[record['Biomarker']]
        if 'FamilyGender' in record:
            numerized['FamilyGender'] = family_gender_map[record['FamilyGender']]
        if 'FamilyDegree' in record:
            numerized['FamilyDegree'] = family_degree_map[record['FamilyDegree']]
        if 'FamilyDiversity' in record:
            numerized['FamilyDiversity'] = record['FamilyDiversity']
        result.append(numerized)

    return result


def json2df(json_ls, cols:list=None):
    df = pd.DataFrame(json_ls)
    if cols is not None:
        df = df[cols]
    return df

def create_grouped_stratified_folds(df, y_column='Biomarker', t_column='SmokerType', n_splits=5, seed=123):
    """
    Create stratified training and testing folds from a DataFrame based on two specified columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        y_column (str): The name of the first column used for stratification.
        t_column (str): The name of the second column used for stratification.
        n_splits (int): Number of folds. Default is 5.
        seed (int): Random seed for reproducibility. Default is 42.
    Returns:
        list of dicts: Each dictionary contains 'train_set' and 'test_set' DataFrames for each fold.
    """
    # Ensure the columns exist in DataFrame
    if y_column not in df.columns or t_column not in df.columns:
        raise ValueError(f"One or both columns: {y_column}, {t_column} not found in DataFrame")

    # Combine the two columns into a single group column
    df['group'] = df[y_column].astype(str) + "_" + df[t_column].astype(str)

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)   # frac=1 to return all rows
    
    sorted_df1 = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    sorted_df2 = df_shuffled.sort_values(by=list(df_shuffled.columns)).reset_index(drop=True)
    print("two dfs are equal: ", sorted_df1.equals(sorted_df2))

    # Initialize the GroupKFold object
    gkf = GroupKFold(n_splits=n_splits)
    
    # Prepare the list to store the folds
    folds = []

    # Generate the indices for each fold using the group column
    for train_idx, test_idx in gkf.split(df_shuffled, groups=df_shuffled['group']):
        # Creating training and testing sets
        train_set = df_shuffled.iloc[train_idx]
        test_set = df_shuffled.iloc[test_idx]
        
        # Append the train and test sets to the list
        train_set = train_set[[col for col in train_set.columns if col != 'group']]
        test_set = test_set[[col for col in train_set.columns if col != 'group']]

        folds.append({
            'train_set': train_set,
            'test_set': test_set
        })
    return folds


"""
    Given a list, where each element has format of 
```{'ehr1': '1006645',
 'age': '55',
 'gender': 'Female',
 'smoking': 'PreviousSmoker',
 'familyType': 'Father',
 'CancerType': 'Prostate',
 'biomarker': 'MET'}```. 
I want to write a function `pre_process` to:
1. replace the keys: 'ehr1', 'age', 'gender', 'smoking', 'familyType', 'CancerType', 'biomarker' with "Patient_id", 'Age', "Gender", "SmokerType",  "Family", "FamilyCancer", "Biomarker".
2. Then categorize 'Age' into two categories: 'Young': if(int(Age) <= 50), 'Old': if (int(Age) > 50)
3. Then categorize 'FamilyCancer' into two categories: 'Major' if 'FamilyCancer' in {Major Cancer: 'Breast', 'Lung', 'Colorrectal', 'Head_and_neck', 'Uterus/cervical', 'Esophagogastric', 'Prostate'} otherwise 'Minor'. 
4. Then categorize 'SmokerType' into three categories: 'Current-Smoker' if 'SmokerType' == 'CurrentSmoker', 'Former-Smoker' if 'SmokerType' == 'PreviousSmoker', 'Never-Smoker' if 'SmokerType' == 'NonSmoker'.
5. Then categorize 'Biomarker' into two categories: 'ALKorEGFR' if 'Biomarker' in {'EGFR', 'ALK'} otherwise 'Others'.
6. For 'Family', replace "UNK", "Father", "Mother", "Brother", "Sister", "Daughter", "Son", "Uncle", "Nephew", "Grandfather", "Grandmother", "Aunt", "Niece", 'Granddaughter', 'Grandson', 'Grandgrandfather', 'Grandgrandmother', "No", 'Halfsister', 'Halfbrother', 'Female_Cousin', 'Male_Cousin', 'NULL' correspondingly by "UNK", "M1", 'F1', 'M1', 'F1', 'F1', 'M1', 'M2', 'M2', 'M2', 'F2', 'F2', 'F2', 'F2', 'M2', 'M3', 'F3', 'No', 'F2', 'M2', 'F3', 'M3', 'NULL'.

Then write another function `per_patient_data` takes the output of the previous function `pre_process` as input, and output another json list:
For each "Patient_id":
1. keep only one value of 'Age', 'FamilyCancer', 'SmokerType' (if has 'Current-Smoker' then 'Current-Smoker' else (if has 'Former-Smoker' then 'Former-Smoker' else 'Never-Smoker')), 'Biomarker' (if has 'ALKorEGFR' then 'ALKorEGFR' else 'Others').
2. remove those json records if with 'Family' is 'UNK', 'NULL', or 'No'.
3. generate 'FamilyGender' if a 'Patient_id' has only 'F[X]'([X] can be 1,2,3) then 'Woman' else (if has only 'M[X]'([X] can be 1,2,3) then 'Man' else 'ManorWoman'). 
4. generate 'FamilyDegree' if a 'Patient_id' has only '[X]1' ([X] can be 'F', 'M') then 'Degree1' else (if has only '[X]2'([X] can be 'F', 'M') then 'Degree2') else (if has only 'UNK' or 'No' then 'Unkown') else 'Degree3'.
5. generate 'FamilyDiversity' as the number of unique 'Family' of a 'Patient_id', but the number should be 3 if the number is >= 3. 
6. remove the key and its values of 'Family'.

Then write another function `digitalize` takes the output of `per_patient_data` as input, and output a json list:
the task is to numerize the values of different attributes of 'Patient_id'. 
1. For 'Age', -1 if 'Young', 1 if 'Old'
2. For 'Gender', -1 if 'Female', 1 if 'Male'
3. For 'SmokerType', 0 if 'Never-Smoker', 2 if 'Current-Smoker', 1 if 'Former-Smoker'
4. For 'FamilyCancer', -1 if 'Minor', 1 if 'Major', 0 if 'MajororMinor'
5. For 'Biomarker', 0 if 'Others', 1 if 'ALKorEGFR'
6. For 'FamilyDegree', 3 if 'Degree3', 2 if 'Degree2', 1 if 'Degree1'
7. For 'FamilyGender', 0 if 'Woman', 1 if 'Man', 2 if 'ManorWoman'
8. For 'FamilyDiversity', keep it.
"""
