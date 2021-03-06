"""Module for Fingerprint"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os.path as osp

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

'''
The code are adapted from
https://github.com/cyrusmaher/ogb-molecule-comp
'''

def smiles2fp(smiles, fp_type):
    r"""Convert smile strings to fingerprint.

    Molecular fingerprints are a way to represent molecules as mathematical objects.

    Note:
        To use this trick, you should install [rdkit](https://www.rdkit.org/) at first:
        ```bash
        pip install rdkit
        ```
    
    Args:
        smiles (list of str): The smile strings to convert.
        fp_type (list of str): The types of generated fingerprint. 
            Can be the following values:

            * `morgan`: Morgan fingerprint.
            * `maccs`: MACCS keys.
            * `rdkit`: RDKit topological fingerprint.
    
    Returns:
        (np.array): The generated fingerprint features.
    """
    fps = {fpt: [] for fpt in fp_type}

    for i in tqdm(range(len(smiles))):
        rdkit_mol = Chem.MolFromSmiles(smiles[i])

        for fpt in fp_type:
            if fpt == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2)
            elif fpt == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(rdkit_mol)
            elif fpt == 'rdkit':
                fp = Chem.RDKFingerprint(rdkit_mol)
            
            fps[fpt].append(fp)

    return [np.array(fps[fpt], dtype=np.int64) for fpt in fp_type]


def ogb2fp(name, root='dataset', fp_type=['morgan', 'maccs']):
    r"""Generate fingerprint features for OGB datasets.

    Molecular fingerprints are a way to represent molecules as mathematical objects.

    Note:
        To use this trick, you should install [rdkit](https://www.rdkit.org/) at first:
        ```bash
        pip install rdkit
        ```
    
    Example:
        [Fingerprint (DGL)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/Fingerprint.ipynb), [Fingerprint (PyG)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/Fingerprint.ipynb)
    
    Args:
        name (str): Name of the dataset.
        root (str): Root directory to store the dataset folder.
        fp_type (list of str, optional): The types of generated fingerprint. 
            Can be the following values:

            * `morgan`: Morgan fingerprint.
            * `maccs`: MACCS keys.
            * `rdkit`: RDKit topological fingerprint.


    Returns:
        (torch.Tensor): The generated fingerprint features.
        (torch.Tensor): The ground truth label.
    """
    smile_path = osp.join(root, name.replace('-', '_'), 'mapping/mol.csv.gz')

    df = pd.read_csv(smile_path)

    print('Converting graphs into fingerprint...')
    fps = smiles2fp(df['smiles'].tolist(), fp_type)

    X = np.concatenate(fps, axis=-1)

    y = df.drop(['smiles', 'mol_id'], axis=1).values
    
    return X, y
        


if __name__ == '__main__':
    ogb2fp('ogbg-molhiv', '/dev/dataset')