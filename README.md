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
    - Contains .csv data files organized by collectoion location. Also contains a .yml file to describe the structure of the files.
- /figures:  
    - Contains the figures in the paper. 
- /src:    
    - /inhibition_induced_devaluation:
        * [main.py]: Script to run the preprocessing and analysis using the utility functions. 
        -/utils: 
            * [utils.py]: Helper functions to condense `main.py` notebooks.  
- /tables:  
    - Contains the tables in the paper.
- /tests:
    - Contains tests for major functions.
    - Run tests using `uv run pytest`
- /output:
    - Contains output stats including ANOVAs, equivalence tests, and Bayes Factors. 