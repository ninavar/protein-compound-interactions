import torchdrug
from torchdrug import data as tdata
from torchdrug import layers
from torchdrug.layers import geometry
from torchdrug import models
import os
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# TorchDrug graph construction model
graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet").to(device)
# Protein embedding model (GearNet-Edge)
gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], 
                              num_relation=7, edge_input_dim=59, num_angle_bin=8,
                              batch_norm=True, concat_hidden=True, short_cut=True, readout="sum").to(device)

pdb_dir = '/dcs/22/u2243582/cs310/gen_graphs/refined_pdbs'
protein_embeddings = {}
count = 0
for pdb_file in os.listdir(pdb_dir):
    if pdb_file.endswith('.pdb'):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        pdb_code = pdb_file.split("_")[0]

        # Load the Protein object from the .pdb fle
        protein = tdata.Protein.from_pdb(pdb_path, atom_feature=None, bond_feature=None)
        protein.view = "residue" # GearNet requires residue graphs
        _protein = tdata.Protein.pack([protein]).to(device)

        # Construct the protein graph
        protein_graph = graph_construction_model(_protein) # [num_nodes, num_nodes, 7]
        node_features = protein_graph.node_feature.float().to(device) # [num_nodes, 21]

        # Use GearNet-Edge to encode protein graph
        with torch.no_grad():
            protein_embedding = gearnet_edge.forward(protein_graph, node_features)
        graph_embedding = protein_embedding["graph_feature"]

        # Save the encoding and the PDB Code
        protein_embeddings[pdb_code] = graph_embedding
        count += 1
        print(count)

torch.save(protein_embeddings, "gearnet_embeddings.pt")

print("Finished saving, now loading the file of ...")
loading = torch.load("gearnet_embeddings.pt")
print(f"{len(loading)} graphs!")