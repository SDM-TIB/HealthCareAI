import pandas as pd
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, XSD

# Function to generate RDF data from DataFrame
def add_patients_to_graph(df, ttl_file_path):
    # Initialize the graph and load the TTL file
    g = Graph()
    g.parse(ttl_file_path, format="turtle")
    
    # Define the namespaces
    nsclcp = Namespace("http://research.tib.eu/clarify2020/vocab#")
    nsclce = Namespace("http://research.tib.eu/clarify2020/entity#")
    
    # Add each patient to the graph
    for idx, row in df.iterrows():
        patient_uri = nsclce[f"patient{idx}"]
        
        # Define the entity type
        g.add((patient_uri, RDF.type, nsclce.NLCPatient))
        
        # Define properties
        if pd.notna(row['Age']):
            g.add((patient_uri, nsclcp.Age, Literal(row['Age'], datatype=XSD.string)))
        if pd.notna(row['Gender']):
            g.add((patient_uri, nsclcp.Gender, Literal(row['Gender'], datatype=XSD.string)))
        if pd.notna(row['FamilyGender']):
            g.add((patient_uri, nsclcp.FamilyGender, Literal(row['FamilyGender'], datatype=XSD.string)))
        if pd.notna(row['FamilyDegree']):
            g.add((patient_uri, nsclcp.FamilyDegree, Literal(row['FamilyDegree'], datatype=XSD.string)))
        if pd.notna(row['SmokerType']):
            g.add((patient_uri, nsclcp.SmokerType, Literal(row['SmokerType'], datatype=XSD.string)))
        if pd.notna(row['FamilyCancer']):
            g.add((patient_uri, nsclcp.FamilyCancer, Literal(row['FamilyCancer'], datatype=XSD.string)))
        if pd.notna(row['Biomarker']):
            g.add((patient_uri, nsclcp.Biomarker, Literal(row['Biomarker'], datatype=XSD.string)))
    
    return g

# Function to summarize RDF graph
def summarize_graph(g):
    num_entities = len(set(subject for subject, _, _ in g.triples((None, RDF.type, None))))
    num_properties = len(set(predicate for _, predicate, _ in g))
    num_triples = len(g)
    
    summary = {
        "Number of entities": num_entities,
        "Number of properties": num_properties,
        "Number of triples": num_triples
    }
    
    return summary

def summarize_kg(df, ttl):
    ttl_file_path = "/Users/jason/Documents/Coding Projects/KPI3/HealthCareAI Web Semantics/causal/meta.ttl"  # Replace with actual path
    g = add_patients_to_graph(df, ttl_file_path)

    # To view the generated RDF data
    # print("Generated RDF data:\n", g.serialize(format="turtle"))
    
    # Summary of the RDF graph
    summary = summarize_graph(g)
    g.serialize(ttl, 'turtle')
    print("\nSummary of RDF graph:\n", summary)

for i in [2000, 5000, 10000]:
    fn = f'/Users/jason/Documents/Coding Projects/KPI3/HealthCareAI Web Semantics/data/{i}_gen.csv'
    ttl = f'/Users/jason/Documents/Coding Projects/KPI3/HealthCareAI Web Semantics/data/{i}_gen.ttl'
    df = pd.read_csv(fn)
    summarize_kg(df, ttl)
