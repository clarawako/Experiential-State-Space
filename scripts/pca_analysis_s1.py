# # PCA analysis
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load cleaned Experience sampling dataset
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Combined_ESM.csv"
print(DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. The data are not publicly available.\n"
    )
else: 
    esm = pd.read_csv(DATA_PATH)

# Find location of Start, finish of thought question columns to be added to the PCA and ID number 
# Print column name and index, ask from which question to which question to filter for pca
print("Column names and their indices:")
for idx, col in enumerate(esm.columns):
    print(f"{col}, Index: {idx}")

# Ask the user for input range (supports single indices, ranges, and combinations like 4,7:20,27)
filtering = input(
    "Please specify the indices of the columns to include (e.g., 6:20 or 4, 7:20, 27): "
).strip()

# Get the location of the "ID" column
try:
    id_loc = esm.columns.get_loc("ID")
except KeyError:
    raise KeyError("The 'ID' column is not found in the dataset. Please check the column names.")

# Parse the input and create a list of indices
selected_indices = []
for part in filtering.split(","):
    if ":" in part:  # Handle ranges like 7:20
        start, finish = map(int, part.split(":"))
        selected_indices.extend(range(start, finish + 1))
    else:  # Handle single indices like 4 or 27
        selected_indices.append(int(part))

# Add the "ID" column index if not already included
if id_loc not in selected_indices:
    selected_indices.append(id_loc)

# Remove rows with missing values in ID column
esm = esm[~esm.loc[:, "ID"].isna()]

# Sort indices to maintain column order
selected_indices = sorted(selected_indices)

# Select the columns based on indices
selected_columns = esm.columns[selected_indices]

# Create the filtered DataFrame
esm_thoughts = esm[selected_columns].dropna()

# Output results
print("Selected columns for PCA:")
print(selected_columns.to_list())
print(f"The resulting dataset contains {esm_thoughts.shape[0]} rows and {esm_thoughts.shape[1]} columns.")

# Count number of responses per participant
number_of_notifications = 42
id_counts = esm_thoughts.groupby('ID').size().reset_index(name='Count')
id_counts["percent response"] = id_counts["Count"]*100 / number_of_notifications

# Filter out participants that responded to less than 40% of all the MDES notifications
filtered_counts = id_counts[id_counts['percent response'] >= 40]

# Get the number of participants that satisfy the condition
sample_size = len(filtered_counts)

# Filter the original dataset to keep only those rows with IDs that replied to more than 40% of notifications sent
esm_filtered = esm_thoughts[esm_thoughts['ID'].isin(filtered_counts["ID"])]

# Standardise the dataframe: centered and scaled
scaler = StandardScaler()

# Drop ID column for standardising and PCA
esm_only = esm_filtered.drop(columns=['ID'])
esm_for_proj = esm_thoughts.drop(columns=['ID'])

# Fit and transform the data
esm_normalised = scaler.fit_transform(esm_only)
esm_proj = scaler.fit_transform(esm_for_proj)

# PCA with varimax rotation, on the normalised data (only participants with more than 40% RR) 
# Apply varimax or not
# Find the number of components to extract
components = input("Do you want to extract a specific number of components? (yes/no): ").strip().lower()
if components in ['yes', 'y']:
    number = input("How many? Please use a numerical value").strip() #Later add more conditions to this, will suffice for now TO DO
    n_components = int(number)
else:
    fa_num = FactorAnalyzer(rotation= None, n_factors=esm_normalised.shape[1], method='principal') #Extract all then only keep those that have an eigenvalue > 1
    fa_num.fit(esm_normalised)
    eigenvalues, fa_eigvalues = fa_num.get_eigenvalues()
    n_components = np.sum(eigenvalues > 1)

apply_rotation = input("Do you want to apply Varimax rotation? (yes/no): ").strip().lower()
if apply_rotation in ['yes', 'y']:
    rotation = 'varimax'
else:
    rotation = None

# Compute the PCA, method = principal is hard coded
fa = FactorAnalyzer(rotation= rotation, n_factors=n_components, method='principal') 
fa.fit(esm_normalised)
eigenvalues, fa_eigvalues = fa.get_eigenvalues()

# Print number of components with eigenvalue larger than 1
print(f"Number of components extracted: {n_components}")
# Scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', color='b')
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.xticks(range(1, len(eigenvalues) + 1))  # Ensure each factor index is labeled
plt.grid()
plt.savefig(f"{ROOT}/results/scree_plot.png")
plt.close()

# Retrieve the factor loadings
loadings = pd.DataFrame(fa.loadings_, columns=[f"Component {x+1}" for x in range(n_components)], index=esm_only.columns[:24])
if rotation:
    print("\nRotated Factor Loadings (Varimax):")
else:
    print("\nUnrotated Factor Loadings:")
print(loadings)

variance, proportional_var, cummulative_var = fa.get_factor_variance()

# Project the whole dataset into the new space
scores = fa.transform(esm_proj)
pca_scores = pd.DataFrame(scores, columns=[f"Component {x+1}" for x in range(n_components)], index = esm_thoughts['ID']) # Shoud be the scores in the new space

print("Variance explained by each factor:", np.round(variance, 2)) #needs to be printed in a nicer way, will do for now TO DO
print("Proportion of variance explained by each factor:", np.round(proportional_var, 3))
print("Cumulative variance explained:", np.round(cummulative_var, 3))

# Features as columns
loadings_ind = loadings.reset_index(names = "Features")
loadings_ind = round(loadings_ind, 2)

save = input("Do you want to save csv's to create Wordclouds? yes/no").strip()
if save in ['yes', 'y']:    
    loadings_wordcloud = round(loadings, 3) * 100

    # Dictionary to store the word cloud DataFrames for each component
    loadings_wc_dict = {}

    # Iterate over component columns (skip the "Features" column if it is the first one)
    for i, column in enumerate(loadings_wordcloud.columns):  # Skip the first column (Features)
        loadings_wc = pd.DataFrame()
        
        # Create the 'weight' column (absolute values of loadings)
        loadings_wc["weight"] = abs(loadings_wordcloud[column].values).astype(int)  # Ensure it's converted to an array
        
        # Create the 'word' column (original features from the index)
        loadings_wc["word"] = loadings_wordcloud.index.tolist()  # Convert index to list
        
        # Create the 'color' column based on the sign of the original value
        loadings_wc["color"] = ["#cc0000" if row > 0 else "#0b5394" for row in loadings_wordcloud[column]]
        
        loadings_wc_dict[column] = loadings_wc.reset_index(drop=True)  # Reset index and drop old one
        
        # Save each DataFrame to a CSV file
        loadings_wc.to_csv(f"{ROOT}/data/loadings_wordcloud_{column}.csv", index=False)

# Save loadings df and pca scores
save = input("Do you want to save the output of the PCA? (yes / no)").strip().lower()
if save in ['yes', 'y']: 
    name = input("What do you want to name your PCA output? Words split by _").strip().lower()  
    pca_scores.to_csv(f"{ROOT}/data/pca_scores_{name}.csv", index= True)## not done more stuff to control for TO DO
    loadings_ind.to_csv(f"{ROOT}/data/pca_loadings_{name}.csv", index= True)

# ### Project Session 2 (emotional manipulation) into this PCA space
# Load session 2 dataset
mood_all = pd.read_csv(f"{ROOT}/data/Combined_Session_2.csv")

# Select same columns as in original PCA but in moods df, drop rows if any NA in selected columns 
mood_subset = mood_all.loc[:, selected_columns].dropna(axis= 0, how = "any")
mood_id = mood_subset["ID"]

# Check column order in session 2 matches column order in session 3 (OG PCA)
if list(mood_subset.columns) == list(esm_filtered.columns):
    print("Columns match")

mood_subset = mood_subset.drop(columns = ["ID"])

# Scale and centre with the same scalers as the original dataset, to maintain consistency 
mood_scaled = scaler.transform(mood_subset)

# Find location of ID, confidence and date, to join the columns in the dataset not included in the PCA to the projected PCA scores later on
id_index = mood_all.columns.get_loc("ID")
task_index = mood_all.columns.get_loc("Task")
conf_index = mood_all.columns.get_loc("Confidence")
date_index = mood_all.columns.get_loc("Date")

# Project mood_scaled to new PCA space
mood_pca = fa.transform(mood_scaled)

# Convert the projected data to a DataFrame 
mood_pca_df = pd.DataFrame(mood_pca, columns=[f"Component_{i+1}" for i in range(mood_pca.shape[1])], index = mood_id)
mood_pca_df.reset_index(inplace=True)

# Select raw thought dimensions and merge onto the projected PCA scores to have one csv with everything.
cols_to_merge = mood_all.columns[id_index : task_index].to_list() + [mood_all.columns[conf_index]] + mood_all.columns[date_index:].to_list()
mood_selected = mood_all.loc[:, cols_to_merge]
mood_selected.reset_index(inplace=True)
mood_projected_full = pd.merge(mood_pca_df, mood_selected, how="inner", left_index= True, right_index= True) # mergig on ID was giving me problems because it was merging many to many and duplicating every row, so joined on index

# Drop index and ID_y columns
mood_projected_full = mood_projected_full.drop(columns=["index", "ID_y"])
mood_projected_full = mood_projected_full.rename(columns= {"ID_x": "ID"})

# Export projected session 2 df
save_results = input("Do you want to save the projected Session 2 dataset? (yes / no)").strip().lower()
if save_results in ['yes', 'y']: 
    name_proj = input("What do you want to name your projected Session 2 output? Words split by _").strip().lower()  
    mood_projected_full.to_csv(f"{ROOT}/data/projected_session_2_scores_{name_proj}.csv")


