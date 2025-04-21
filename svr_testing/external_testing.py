import pandas as pd
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, EsmTokenizer, EsmModel
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained models
esm = "facebook/esm2_t12_35M_UR50D" # generate protein embeddings
esm_tokenizer = EsmTokenizer.from_pretrained(esm)
esm_model = EsmModel.from_pretrained(esm).to(device)
chemberta = "DeepChem/ChemBERTa-10M-MTR" # generate ligand embeddings
chemberta_tokenizer = AutoTokenizer.from_pretrained(chemberta)
chemberta_model = AutoModel.from_pretrained(chemberta).to(device)
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
svr = joblib.load("svr_model.pkl")

# Load the genera-testing-set.csv
test = pd.read_csv("/dcs/22/u2243582/cs310/feature_extraction/general-testing-set.csv")
test_y = test['Log_binding'].tolist()
test_fasta = test['Protein_FASTA'].tolist()
test_smile = test['Ligand_SMILES'].tolist()

# Batch processing for protein sequences
def generate_protein_embeddings(fasta_sequences, batch_size=4, max_length=1024):
    esm_model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(fasta_sequences), batch_size), desc="Processing Protein Batches"):
            batch = fasta_sequences[i:i + batch_size]
            inputs = esm_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            outputs = esm_model(**inputs)
            # Extract the last layer and apply mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)
            # Clear GPU memory after processing the batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.concatenate(all_embeddings, axis=0)

# Batch processing for ligand SMILES strings
def generate_ligand_embeddings(smiles_list, batch_size=4):
    chemberta_model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Processing Ligand Batches"):
            batch = smiles_list[i:i + batch_size]
            inputs = chemberta_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = chemberta_model(**inputs)
            # Extract the last layer and apply mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)
            # Clear GPU memory after processing the batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.concatenate(all_embeddings, axis=0)

# Generate Embeddings
protein_embeddings = generate_protein_embeddings(test_fasta)
ligand_embeddings = generate_ligand_embeddings(test_smile)
embeddings = np.concatenate((protein_embeddings, ligand_embeddings), axis=1) # (5059, 864)
test_X = scaler.transform(embeddings)
test_X = pca.transform(test_X)
yPred = svr.predict(test_X)

pearsonCoef, _ = pearsonr(test_y, yPred)
print("Pearson Correlation Coefficient:", pearsonCoef)

spearmanCoef, _ = spearmanr(test_y, yPred)
print("Spearman Correlation Coefficient:", spearmanCoef)

mae = mean_absolute_error(test_y, yPred)
print("Mean Absolute Error:", mae)

variance = np.var(np.array(test_y) - np.array(yPred))
print("Variance of Errors:", variance)

r2 = r2_score(test_y, yPred)
print("R2 Score:", r2)

# Create hexbin plot
plt.figure(figsize=(8, 6))
hb = plt.hexbin(test_y, yPred, gridsize=35, cmap='plasma', mincnt=1)
# Add a colorbar to indicate density
cb = plt.colorbar(hb)
cb.set_label('Number of examples')
# Add labels and title
plt.xlabel("True Binding Affinity")
plt.ylabel("Predicted Binding Affinity")
plt.title("SVR + sequence embeddings True vs Predicted values on PDBbind general-set")
# Add a diagonal line y = x to show the ideal prediction
min_val = min(min(test_y), min(yPred))
max_val = max(max(test_y), max(yPred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
plt.legend()
plt.savefig("scatter_true_vs_predicted.png", dpi=300)  # save to file
plt.show()