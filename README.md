# DS_340W_Course_Project

This project aims to accurately classify malicious user in a specific category (By tag) on a social media platform through graph neural network (possibly sentiment analysis) and presents data in a meaningful way. 

## Steps:

1) Gather historical data (Labeled).
2) Define malicious user and determine the validity of the standard.
3) Exploratory data analysis using the existing data, including any need for feature engineering.
4) Train/Test set using GNN and other classification algorithm.
5) Classify newly scrape data.
6) Build an interactive dashboard in the resulting data.

## Potential Problems:

- We need to determine the validity of our classification result. Which can be achieve by analyze variable importance, and create potential criterions. Criterions can be created automatically using a decision tree.

## Path on Bridge2 (And some instruction)

- cd $Project/ocean/projects/cis240116p/username
- module load anaconda3
- conda activate env (make sure env is created)
- sbatch batch
- squeue -u user_name
