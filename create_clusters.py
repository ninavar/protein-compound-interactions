# ! Using code from https://github.com/foxtrotmike/BioTools/blob/main/processCDHIT.py

#conda install -c conda-forge biopython
#conda install -c bioconda cd-hit
# conda activate myenv

import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import json

def processCDHIT(L,cthresh=0.4,ofile="out.cdhit"):
    """   
    Generate CD-HIT clustering
    Runs CD-HIT and creates a temporary file which is not deleted automatically
    Parameters
    ----------
    L : TYPE Fasta file string OR List of protein sequences OR  SeqRecord
        DESCRIPTION.
    cthresh : TYPE, optionals
        DESCRIPTION. Cutoff threshold The default is 0.8.
    ofile : TYPE, optional
        DESCRIPTION. The default is "out.cdhit".

    Returns
    -------
    cc : TYPE Dictionary with cluster id string as key and list of protein ids in each cluster
        DESCRIPTION. 

    """
    # Choose of word size:
    #   -n 5 for thresholds 0.7 ~ 1.0
    #   -n 4 for thresholds 0.6 ~ 0.7
    #   -n 3 for thresholds 0.5 ~ 0.6
    #   -n 2 for thresholds 0.4 ~ 0.5

    if type(L)==type(""):
        ifile = L
    else:
        if type(L[0]==type("")):
            L = [SeqRecord(Seq(p),id=str(i)) for i,p in enumerate(L)]
        ifile = ofile+"_temp.fasta"
        with open(ifile, "w") as output_handle:
            SeqIO.write(L, output_handle, "fasta") 
    
    cmd = "cd-hit -i "+ifile+" -d 0 -o "+ofile+" -c "+str(cthresh)+" -n 2  -G 1 -g 1 -b 20 -l 10 -s 0.0 -aL 0.0 -aS 0.0 -T 4 -M 32000"   
    os.system(cmd)
    with open(ofile+".clstr","r") as fh:
        clusters = fh.readlines()
    cc = {}
    for x in clusters:
        xs = x.split()
        if xs[0]=='>Cluster':
            ccid = xs[1]
            cc[ccid]=[]
        else:
            pid = xs[2][1:].split('...')[0]
            cc[ccid].append(pid)            
    return cc

#fastas = "/dcs/22/u2243582/cs310/nrkf_physicochem_feat/pdbcodes.fasta"
fastas = "/dcs/22/u2243582/cs310/gen_graphs/gearnetpdb.fasta"
clusterss = processCDHIT(fastas)
# Writing dictionary to JSON file
with open("clustered_proteins.json", "w") as json_file:
    json.dump(clusterss, json_file)