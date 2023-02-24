# LAGCN
> Code and Datasets for "Predicting Drug-Disease Associations through Layer Attention Graph Convolutional Networks" https://doi.org/10.1093/bib/bbaa243
## Datasets
- data/drug_dis.csv is the drug_disease association matrix, which contain 18416 associations between 269 drugs and 598 diseases.

- data/drug_sim.csv is the drug similarity matrix of 269 diseases,which is calculated based on drug target features.

- data/dis_sim.csv is the disease similarity matrix of 598 diseases,which is calculated based on disease mesh descriptors.
## Code
### Environment Requirement
The code has been tested running under Python 3.6.8. The required packages are as follows:
- numpy == 1.15.4
- scipy == 1.1.0
- tensorflow == 1.12.0
### Usage
```shell
git clone https://github.com/storyandwine/LAGCN.git
cd LAGCN/code
python main.py
```
