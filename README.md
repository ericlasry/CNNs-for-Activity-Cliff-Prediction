# CNNs for AC Prediction.
Exploration and analysis of CNN architectures for Activity Cliff prediction.

## Contents
--------------

* ### Notebooks: 
>#### Section 0: Data processing and exploratory analysis
>>  - 0.0: Draws fragmented image MMP representations from SMILES
>>  - 0.1: Splits and saves image data using the procedure below
>>  - 0.2: Generates 498-bit MACCS fingerprint-based MMP representations from SMILES
>>  - 0.3: EDA for the factor Xa inhibitor dataset
>
>#### Section 1: Training CNNs and AutoGluon classifiers
>>  - 1.0: Training procedure for CNNs
>>  - 1.1: Training AutoGluon models
>
>#### Section 2: Evalution of trained models
>>  - 2.0: CNN evaluation metrics
>>  - 2.1: AutoGluon evaluation metrics
>>  - 2.2: ROC curves for all models
>
>#### Section 3: Explainability methods
>>  - 3.0: Grad-CAM for CNN predictions
>>  - 3.1: SHAP for AutoGluon predictions

* ### superpac
> - base.py : 
> - rdkit.py : 
> - eval.py : 
> - gradcam.py : 

* ### data


* ### metrics
> - metrics_cnn :
> - metrics_ag :

* ### models
> - agClassifier :

* ### misc
> - ARC_scripts :
> - tex_tables : 


## Raw Data Generation
--------------
I used the mmpdb package to scan for MMPs. Note that an MMP can have several non-unique decompositions into a core and two variable parts. To establish uniqueness, for each MMP, I only kept the version with the core with the largest number of heavy atoms. MMPs were labelled as ACs if their pKi absolute difference was larger than two and labelled as non-ACs if the pKi absolute difference was less than one. I removed the "half-cliffs" which had a pKi absolute difference between 1 and 2 (only about 10-15 \%) of MMPs were half-cliffs if I remember correctly). This cleaning-up procedure left me with a well-defined binary classification problem, which included 15787 MMPs, 1862 of which were ACs and the rest were non-ACs

- x_smiles = list of 3435 SMILES strings
- y = list of pKi labels associated with x_smiles
- X_smiles_mmps = all 15787 clean MMPs which can be found in the molecular space defined by x_smiles and which are either non-cliffs or cliffs (no half-cliffs included)
- X_smiles_vcvs = same as X_smiles_mmps, but here the MMP is represented via first variable part + structural core + second variable part instead of two SMILES strings
- y_mmps = binary labels for X_smiles_mmps indicating presence/absence of an activity cliff
- Y_mmps_vals = original numerical pKi values for X_smiles_mmps (has the same shape as X_smiles_mmps)


## MMP Data Splitting Procedure
--------------
Splitting up MMPs in a useful way for machine learning is less trivial than one might think intitially. I experimented with several types of data splits (the literature on MMP data splits is a mess in my opinion). Ultimately I found it most useful to first conduct a train/test split at the level of individual molecules (as usual) and then use this split as the basis to further split up the MMPs in the molecular space into different groups:

- zero_out: the MMPs where both compounds are in the (underlying, single-molecule) training set
- one_out: the MMPs where exactly one compound is in the (underlying, single-molecule) test set
- two_out: the MMPs were both compounds are in the (underlying, single-molecule)test set
- two_out_seen_core: same as two_out, but only with structural cores which appear in the (underlying, single-molecule) training set
- two_out_unseen_core: same as two_out, but only with structural cores which do not appear in the (underlying, single-molecule) training set


Index sets:
- x_smiles[ind_train_mols] = all smiles in the (underlying, single-molecule) training set
- x_smiles[ind_test_mols] = all smiles in the (underlying, single-molecule) test set
- X_smiles_mmps[ind_two_out_mmps] = All MMPs where both compounds lie in x_smiles[ind_test_mols]
- X_smiles_mmps[ind_one_out_mmps] = All MMPs where exactly one compound lies in x_smiles[ind_train_mols] and the other compound lies in x_smiles[ind_test_mols]
