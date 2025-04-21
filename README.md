# protein-compound-interactions

GitHub repository for project "Developing Machine Learning Models for Protein and Compound Interactions".

The FASTA and SMILES strings for the complexes of the refined-set can contained in data/refined-set.csv. The file contains 5250 complexes - a further 183 complexes need to be removed due to containing 'X' in FASTA string (ambiguous char.), and 8 complexes are duplicates under different experimental conditions. These are preprocessed for each experiment (would have been more efficient to remove from .csv though).

Xtestfeatnew.csv and Xtrainfeatnew.csv contain the physicochemical features of the refined-set. These were originally split into test and train dataset, before the final experiment design, so are preprocessed into the same DataFrame for each experiment - similarly for the binding affinity values in ytest.csv and ytrain.csv. Again, would be more efficient to combine back into single files, however, these experiments were conducted during the 'learning-curve' phase of the project, getting used to data preprocessing and handling...

ChemBERTA compound embeddings saved in ligand_embeddings_t12.npy. ESM2 protein embeddings saved in protein_embeddings_t12.py. File of compressed graphs > 25MB so included code to generate graphs, and generate GearNet embeddings, and .zip of GearNet embeddings.

Independent test set curated from PDBbind general-set saved in general-testing-set.csv.
