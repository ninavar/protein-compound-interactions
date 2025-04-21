import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from transformers import AutoTokenizer, AutoModel, EsmTokenizer, EsmModel
# from tqdm import tqdm
import joblib

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the training data
data = pd.read_csv('/dcs/22/u2243582/cs310/feature_extraction/refined-set-csv.csv')
data = data[~data['Protein_FASTA'].str.contains('X')] # remove complexes with ambiguous char in FASTA string
toDrop = ['4yx4', '1laf', '4buq', '1k22', '1hmt', '1utn', '4rux', '1bty']
data = data[~data['PDB_Code'].isin(toDrop)]
data = data.reset_index() # Now data is the same shape and order as embeddings

testSet = ['5i3a', '2p15', '3g30', '6pg6', '4qfo', '1jak', '4rqk', '1zs0', '5am6', '3q6w', '3qxv', '4xtv', '2x00']
indices = data.index[data['PDB_Code'].isin(testSet)]
"""
index PDB_Code                                      Protein_FASTA                                      Ligand_SMILES Binding_data  Log_binding
426     455     1jak  DRKAPVRPTPLDRVIPAPASVDPGGAPYRITRGTHIRVDDSREARR...      CC(=O)N[C@H]1[NH2+]C[C@H](CO)[C@H](O)[C@@H]1O     Ki=2.7uM         5.57
513     543     1zs0  MLTPGNPKWERTNLTYRIRNYTPQLSEAEVERAIKDAFELWSVASP...  COc1ccc(-c2ccc(S(=O)(=O)N[C@H](C(C)C)[P+](=O)(...     Ki=700nM         6.15
1182   1248     3g30  TSAVQQKLAALEKSSGGRLGVALIDTADNTQVLYRGDERFPMCSTS...           O=C([O-])[C@H]1C[C@H]1C(=O)Nc1cc(F)ccc1F     Ki=3.1mM         2.51
1662   1750     6pg6  VKPNYALKFTLAGHTKAVSSVKFSPNGEWLASSSADKLIKIWGAYD...                          CC(=O)NCCCCc1nc(CO)c[nH]1     Kd=200uM         3.70
1801   1894     4xtv  TDRDRLRPPLDERSLRDQLIGAGSGWRQLDVVAQTGSTNADLLARA...  Nc1ncnc2c1ncn2[C@H]1CC[C@@H](CNS(=O)(=O)NC(=O)...    Kd=0.54nM         9.27
1947   2043     2x00  YKDDDDKLHSQANLMRLKSDLFNRSPMYPGPTKDDPLTVTLGFTLQ...  CC1=C([C@@H]2C[C@H](C)C(=O)O2)CC[C@]23CCCN=C2C...     Kd=4.7pM        11.33
2424   2528     3qxv  VQLVESGGGLVQAGGSLRLSCAASRRSSRSWAMAWFRQAPGKEREF...  CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CC...       Kd=4nM         8.40
2554   2661     2p15  SLALSLTADQMVSALLDAEPPILYSEYRPFSEASMMGLLTNLADRE...  C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C...      Kd=50pM        10.30
3699   3838     4qfo  QGLVYCAEANPVSFNPQVTTTGSTIDIIANQLYDRLISIDPVTAEF...     CSCC[C@H]([NH3+])C(=O)N[C@@H](CC(C)C)C(=O)[O-]    Kd=53.4uM         4.27
4296   4460     5am6  VAGVSEYELPEDPRWELPRDRLVLGKPLGEGQVVLAEAIGLDKDKP...  C[NH+]1CCN(c2ccc3nc(-c4c(N)c5c(F)cccc5[nH]c4=O...     Kd=185nM         6.73
4937   5123     5i3a  KYRVRKNVLHLTDTEKRDFVRTVLILKEKGIYDRYIAWHGAAGKFH...                                       Oc1ccc(O)cc1       Kd=9uM         5.05
"""

final_data = data[~data['PDB_Code'].isin(testSet)]
y = final_data['Log_binding'].to_numpy()

protein_emb = np.load("/dcs/22/u2243582/cs310/seq_embeddings/protein_embeddings_t12.npy")
ligand_emb = np.load("/dcs/22/u2243582/cs310/seq_embeddings/ligand_embeddings_t12.npy")

# Single array of embeddings data
embeddings = np.concatenate((protein_emb, ligand_emb), axis=1) # (5059, 864)
test_embeddings = embeddings[indices] # save the testSet embeddings for testing
embeddings = np.delete(embeddings, indices, axis=0) # remove the testSet complexes to get (5046, 864)

# Scale the data and save the model
scaler = StandardScaler()
scaler.fit(embeddings)
X = scaler.transform(embeddings)
joblib.dump(scaler, "scaler.pkl")

# Apply PCA and save the model
pca = PCA(0.85) # keep 80% of the variance of the data
pca.fit(X)
X = pca.transform(X)
joblib.dump(pca, "pca.pkl")
print ("Components:", pca.n_components_ , "Total explained variance:", pca.explained_variance_ratio_.sum())

# Verify the model through 5 fold CV
pearsonCoeffs, spearmanCoeffs, maeScores, varianceScores, r2Scores= [], [], [], [], []

cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)

for train_ix, test_ix in cv_outer.split(X):

    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]

    model = SVR(kernel='rbf', gamma='scale', epsilon=0.6, C=8)
    model.fit(X_train, y_train)
    yPred = model.predict(X_test)

    pearsonCoef, pearsonP = pearsonr(y_test, yPred)
    pearsonCoeffs.append(pearsonCoef)
    
    spearmanCoef, spearmanP = spearmanr(y_test, yPred)
    spearmanCoeffs.append(spearmanCoef)
    
    mae = mean_absolute_error(y_test, yPred)
    maeScores.append(mae)
    
    variance = np.var(y_test - yPred)
    varianceScores.append(variance)

    r2 = r2_score(y_test, yPred)
    r2Scores.append(r2)
    
metricsSummary = {
    "Pearson Correlation": (np.mean(pearsonCoeffs), np.std(pearsonCoeffs)),
    "Spearman Correlation": (np.mean(spearmanCoeffs), np.std(spearmanCoeffs)),
    "Mean Absolute Error": (np.mean(maeScores), np.std(maeScores)),
    "Variance of Errors": (np.mean(varianceScores), np.std(varianceScores)),
    "R2": (np.mean(r2Scores), np.std(r2Scores))}
for metric, (mean, std) in metricsSummary.items():
    print(f"{metric}: Mean = {mean}, Std = {std}")

# Train final model and save for deployment
svr_model = SVR(kernel='rbf', gamma='scale', epsilon=0.6, C=8)
svr_model.fit(X, y)
joblib.dump(svr_model, "svr_model.pkl", compress=3)
print("saved the model") # Shouldn't need to run this program again...

# ----------------------------------------------------------------------------------------------------------------------
# Testing it on the complexes...
testSet = data[data['PDB_Code'].isin(testSet)]
print(testSet)
testX = scaler.transform(test_embeddings)
testX = pca.transform(testX)
testy = testSet['Log_binding'].to_numpy()
yPred = model.predict(testX)

print(pearsonr(testy, yPred))
print(spearmanr(testy, yPred))
print(mean_absolute_error(testy, yPred))
print(np.var(testy - yPred))
print(r2_score(testy, yPred))

"""
Using device: cpu
Components: 65 Total explained variance: 0.85159826
Pearson Correlation: Mean = 0.7393495013801428, Std = 0.013146015486124802
Spearman Correlation: Mean = 0.7351577340335493, Std = 0.015156757913245927
Mean Absolute Error: Mean = 1.0137544416712096, Std = 0.01911490196154471
Variance of Errors: Mean = 1.727692401179383, Std = 0.09107298329952913
R2: Mean = 0.5433813890364294, Std = 0.020213835862047606
saved the model
...
PearsonRResult(statistic=np.float64(0.8093384992492362), pvalue=np.float64(0.002544652540023648))
SignificanceResult(statistic=np.float64(0.881818181818182), pvalue=np.float64(0.0003301688021839005))
1.3323770293234352
2.553438735928456
0.6478716315498407
"""