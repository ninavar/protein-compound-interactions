import os
import torch
import numpy as np
import pickle
import Bio
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from Bio.PDB.Polypeptide import PPBuilder, is_aa
from rdkit import Chem
import time

# Source genenv/bin/activate

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
one_hot = {aa: torch.eye(len(amino_acids))[i] for i, aa in enumerate(amino_acids)}

# To replace Bio.PDB.Polypeptide.three_to_one()
def amino_shorten(r):
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    return d[r]

def parse_protein(proteinpdb):
    # Parse the protein graph
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", proteinpdb)
    residues, torsions = [], []

    for chain in structure:
        polypeptides = PPBuilder().build_peptides(chain)
        for poly in polypeptides:
            phi_psi = poly.get_phi_psi_list()
            for i, res in enumerate(poly):
                if is_aa(res, standard=True):
                    residues.append(res)
                else:
                    # print("Protein contains invalid X amino acid")
                    return None 
                    
                phi, psi = phi_psi[i]
                if phi is None: phi = 0
                if psi is None: psi = 0
                torsions.append((phi, psi))
    alpha_carbons = [atom for res in residues for atom in res.get_atoms() if atom.get_name() == "CA"]
    # Build a KDTree for efficient neighbour searching
    positions = torch.tensor(np.array([atom.get_coord() for atom in alpha_carbons]))
    kdtree = cKDTree(positions.numpy())
    # Find neighbours within 10 angstroms
    distances, neighbors = kdtree.query(positions, k=9, distance_upper_bound=10.0)
    x, y = [], []

    chains = set([r.get_parent() for r in residues])
    phi_psi = dict(zip(chains,[PPBuilder().build_peptides(c)[0].get_phi_psi_list() for c in chains]))
    for i, res in enumerate(residues):
        # Get the torsion angles (phi and psi)
        phi,psi = torsions[i]
        # Convert torsion angles to radians and add to node features
        torsion_angles = torch.tensor([phi, psi], dtype=torch.float32)
        #x.append(torch.cat([one_hot[res.get_parent().get_resname()], torsion_angles]))
        encoding = one_hot[amino_shorten(res.get_resname())]
        x.append(encoding)          
        y.append(torsion_angles)
    
    # Create node features and edge indices
    x,y = torch.stack(x), torch.stack(y)
    edge_index, edge_distance = [], []
    for i, neighbor_indices in enumerate(neighbors):
        vidx = np.nonzero(neighbor_indices != len(residues))[0]
        valid_neighbors = neighbor_indices[vidx]  # Exclude invalid indices
        edge_index.extend([(i, n) for n in valid_neighbors])
        edge_distance.extend([distances[i, j] for j in vidx])

    # Create the PyTorch Geometric Data object
    data = Data(x=x, edge_index=torch.tensor(edge_index).t().contiguous(),y=y,coords=positions,edge_attr=torch.tensor(edge_distance))
    return data

def parse_ligand(mol):
    atoms = mol.GetAtoms()
    x = torch.tensor([atom.GetAtomicNum() for atom in atoms], dtype=torch.float32).view(-1, 1)

    bonds = mol.GetBonds()
    edge_index, edge_attr = [], []

    for bond in bonds:
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_attr.append([bond.GetBondTypeAsDouble()])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def generate_graphs(root_dir, output_file):
    count = 0
    start_time = time.time()

    final_graphs = {}
    for subdir, dirs, files in os.walk(root_dir):

        subdir_name = os.path.basename(subdir)
        print(f"In subdir {subdir_name}")
        count +=1 

        proteinpdb = None
        ligandmol2 = None

        for file in files:
            if file.endswith(".mol2"):
                ligandmol2 = os.path.join(subdir, file)
            elif "protein" in file and file.endswith(".pdb"):
                proteinpdb = os.path.join(subdir, file)

        if proteinpdb and ligandmol2:
            protein_graph = parse_protein(proteinpdb)

            mol = Chem.MolFromMol2File(ligandmol2, sanitize=True)
            if mol:
                ligand_graph = parse_ligand(mol)
            else:
                print(f"Didn't load ligand molecule from file {subdir_name}")
                continue

            if protein_graph and ligand_graph:
                final_graphs[subdir_name] = [protein_graph, ligand_graph]
                print(f"Successfully generated graphs for {subdir_name}")
            else:
                if not protein_graph:
                    print(f"Protein contained X {subdir_name}")
                if not ligand_graph:
                    print(f"Idk how we got here {subdir_name}")

        else:
            if not proteinpdb:
                print(f"Didn't load protein file {subdir_name}")
                continue
            if not ligandmol2:
                print(f"Didn't load ligand file {subdir_name}")
                continue
    torch.save(final_graphs, output_file)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"Generated {len(final_graphs)} graphs")
    print(f"Walked {count} files")

# Directories
root_dir = "/Users/nina/pbd 2020 data/refined-set"
output_file = "3dprotein_2dligand_graphs.pkl"
generate_graphs(root_dir, output_file)