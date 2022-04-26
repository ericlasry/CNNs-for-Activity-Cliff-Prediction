### Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from PIL import Image




def draw_frag_mmps_from_smiles(df):
    """ Draw fragmented MMPs from SMILES strings. 

    Parameters
    ----------
    df: A pandas DataFrame with a row for each MMP with columns as SMILES of var1, core, and var2 fragments

    Returns
    -------
    mmp_im_list: List of images of pairs - horizontally concatenated images of fragments (core, var1, var2)
    """
    IMAGE_WIDTH = 200
    IMAGE_HEIGHT = 200
    
    mmp_im_list = []
    for i in tqdm(range(len(df))): # Loop through MMPs
        # Store fragments
        var1 = Chem.MolFromSmiles(df.at[i,0])
        core = Chem.MolFromSmiles(df.at[i,1])
        var2 = Chem.MolFromSmiles(df.at[i,2])
        # Draw fragments separately
        var1_im = Chem.Draw.MolToImage(var1).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        core_im = Chem.Draw.MolToImage(core).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        var2_im = Chem.Draw.MolToImage(var2).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        # Put it together
        im = Image.new('RGB', (600,200))
        im.paste(core_im, (0,0))
        im.paste(var1_im, (IMAGE_WIDTH,0))
        im.paste(var2_im, (IMAGE_WIDTH*2,0))

        mmp_im_list.append(im)
    return mmp_im_list


def draw_aug_mmps_from_smiles(df, angle_arr, labels, target_folder):
    """ Draw a rotationally augmented set of MMPs from SMILES strings and save to target_folder.

    Parameters
    ----------
    df: A pandas DataFrame (n rows) with a row for each MMP with columns as SMILES of each molecule in the MMP
    angle_arr: An nx(n_aug)x3 numpy array with (i,j,k)th entry as the rotation applied to kth PART of jth AUGmentation of ith MMP in degrees
    labels: An n-list with classifications of the MMPs as AC (1) or non-AC (0)
    target_folder: A string specifying the path of the folder where the images will be saved.

    Returns
    -------
    None
    """
    IMAGE_WIDTH = 200
    IMAGE_HEIGHT = 200

    if (angle_arr.shape[2] != 3) or (angle_arr.shape[0] != len(df)):# Sanity checks:
        raise ValueError('Check arrays')

    blank = Image.new('RGBA', (IMAGE_WIDTH,IMAGE_HEIGHT), 'white')

    for i in tqdm(range(angle_arr.shape[0])):# Loop through MMPs
        # Store MMP fragments and respective label
        var1 = Chem.MolFromSmiles(df.at[i,0])
        core = Chem.MolFromSmiles(df.at[i,1])
        var2 = Chem.MolFromSmiles(df.at[i,2])

        var1_im = Chem.Draw.MolToImage(var1).resize((IMAGE_WIDTH,IMAGE_HEIGHT))
        core_im = Chem.Draw.MolToImage(core).resize((IMAGE_WIDTH,IMAGE_HEIGHT))
        var2_im = Chem.Draw.MolToImage(var2).resize((IMAGE_WIDTH,IMAGE_HEIGHT))

        mol_im_list = [var1_im, core_im, var2_im]

        label = labels[i]
        
        for j in range(angle_arr.shape[1]):# Loop through augmentations
            arr_list = []
            
            for k in range(angle_arr.shape[2]):# Loop through fragments of the MMP
                # Create clean rotated image of each fragment
                sub_im = mol_im_list[k]
                sub_im = sub_im.convert('RGBA').rotate(angle_arr[i,j,k])
                n_sub_im = Image.composite(sub_im, blank, sub_im).convert('RGB')
                
                sub_arr = np.asarray(n_sub_im)
                arr_list.append(sub_arr)
            
            arr = np.concatenate(arr_list, axis=1) # Concatenate parts into whole image and add to aug_im_list for ith MMP
            im = Image.fromarray(arr).resize((3*IMAGE_WIDTH,IMAGE_HEIGHT))

            name = 'mmp_'+'{:05}'.format(i)+'_'+'{:02}'.format(j)+'.jpg'
            im = im.save(target_folder+"/"+str(label)+'/'+name) # Save MMP images to relevant class folder
    return None


def draw_mmps_from_smiles(df):
    """ Draw mol+mol MMPs from SMILES strings. 

    Parameters
    ----------
    df: A pandas DataFrame with a row for each MMP with columns as SMILES of each molecule in the MMP

    Returns
    -------
    mmp_im_list: List of images of pairs - horizontally concatenated images of both molecules
    """
    IMAGE_WIDTH = 300
    IMAGE_HEIGHT = 300

    mmp_im_list = []
    for i in tqdm(range(len(df))):
        m1 = Chem.MolFromSmiles(df.at[i,0])
        m2 = Chem.MolFromSmiles(df.at[i,1])

        m1_im = Chem.Draw.MolToImage(m1).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        m2_im = Chem.Draw.MolToImage(m2).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        
        im = Image.new('RGB', (2*IMAGE_WIDTH, IMAGE_HEIGHT))
        im.paste(m1_im, (0,0))
        im.paste(m2_im, (IMAGE_WIDTH,0))

        mmp_im_list.append(im)
    return mmp_im_list


def get_fingerprint_encoding_df(smiles):
    """ Obtain MMP MACCS encoding (C, F1, F2) from SMILES strings. 

    Parameters
    ----------
    smiles: A pandas DataFrame with a row for each MMP with columns as SMILES of each fragment in the MMP 

    Returns
    -------
    df: A pandase DataFrame containing bits of encodings for each MMP
    """
    rows = []
    for i in tqdm(range(len(smiles))):
        core = Chem.MolFromSmiles(smiles.at[i,1])
        F1 = Chem.MolFromSmiles(smiles.at[i,0])
        F2 = Chem.MolFromSmiles(smiles.at[i,2])

        maccs_core = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(core)
        maccs_F1 = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(F1)
        maccs_F2 = Chem.GetMACCSKeysFingerprint(F2)

        core_string = maccs_core.ToBitString()[1:]
        F1_string = maccs_F1.ToBitString()[1:]
        F2_string = maccs_F2.ToBitString()[1:]

        core_vec = [int(char) for char in core_string]
        F1_vec = [int(char) for char in F1_string]
        F2_vec = [int(char) for char in F2_string]

        row_vec = core_vec + F1_vec + F2_vec
        
        for c,v in enumerate(row_vec):
            if v not in [0,1]: # Check for non-binary
                raise ValueError("Fingerprints must be binary")

        rows.append(row_vec)

    col_names = []
    for fragment in ['core ', 'F1 ', 'F2 ']:
        for i in range(166): # 166 bits in each MACCS fingerprint
            name = fragment+str(i+1)
            col_names.append(name)

    df = pd.DataFrame(rows, columns=col_names)

    return df

