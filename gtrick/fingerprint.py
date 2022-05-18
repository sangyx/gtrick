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
    fps = {fpt: [] for fpt in fp_type}

    for i in tqdm(range(len(smiles))):
        rdkit_mol = Chem.MolFromSmiles(smiles[i])

        for fpt in fp_type:
            if fpt == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2)
            elif fpt == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(rdkit_mol)
            elif fpt == 'rdit':
                fp = Chem.RDKFingerprint(rdkit_mol)
            
            fps[fpt].append(fp)

    return [np.array(fps[fpt], dtype=np.int64) for fpt in fp_type]


def ogb2fp(name, root='dataset', fp_type=['morgan', 'maccs']):
    smile_path = osp.join(root, name.replace('-', '_'), 'mapping/mol.csv.gz')

    df = pd.read_csv(smile_path)

    print('Converting graphs into fingerprint...')
    fps = smiles2fp(df['smiles'].tolist(), fp_type)

    X = np.concatenate(fps, axis=-1)

    y = df.drop(['smiles', 'mol_id'], axis=1).values
    
    return X, y
        


if __name__ == '__main__':
    ogb2fp('ogbg-molhiv', '/dev/dataset')