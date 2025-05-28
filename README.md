# InhibitionInducedDevaluation

## Description
This project is looking to see whether inhibition induced devaluation occurs in three direct registered replications of Wessel and colleagues (2014).

## Installation
Clone the repository using:

```bash
git clone https://github.com/kriti-ach/InhibitionInducedDevaluation.git
```

Go into the repo using:

```bash
cd /path/to/InhibitionInducedDevaluation
```

Set up the environment using:
```bash
source setup_env.sh
```

## Repository Structure

- /data:  
    - Contains .csv data files organized by collection location. Also contains a .yml file to describe the structure of the files.
- /figures:  
    - Contains the figures in the paper. 
- /src:    
    - /inhibition_induced_devaluation:
        * [main.py]: Script to run the preprocessing and analysis using the utility functions. 
        - /utils: 
            * [utils.py]: Helper functions to condense `main.py` notebooks.
            * [globals.py]: Contains the global variables, including the explicit knowledge subjects from each location. 
            * [phase2_utils.py]: Contains utility functions to process Phase 2.
            * [phase3_utils.py]: Contains utility functions to process Phase 3.
- /tables:  
    - Contains the tables in the paper.
- /tests:
    - Contains tests for major functions.
    - Run tests using `uv run pytest`
- /output:
    - Contains output stats including Bayes Factors and exclusion summaries. 

## Notes

This repository used the Cookiecutter template from: https://github.com/lobennett/uv_cookie

## Notes about OSF Repos

The following OSF repositories have previous work relevant to this manuscript. We initially included 2 Direct Replications (DRs) and 7 Conceptual Replications (CRs) in our Registered Replication. However, in our final manuscript, we did not include data from the 7 CRs as their procedures differed quite a bit from Wessel et al. (2014) and they had small sample sizes. The DRs and CRs are referenced in the OSF repos below. 

- Inhibition Induced Devaluation Registered Replication (https://osf.io/x38aj/)
    - This repository has data from the 2 DRs and 7 CRs. 
    - It also has the Registered Replication pdf document and the scripts used to analyze the data. 

- Inhibition Induced Devaluation Replication (https://osf.io/kq5xd/)
    - This repository has a power analysis to determine sample size for our DRs and CRs. 