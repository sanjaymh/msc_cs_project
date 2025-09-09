## AI-driven PR Reviewer Suggestion System

This project is part of an MSc dissertation. It implements a reviewer recommendation system for GitHub pull requests using a combination of:

Heuristic methods (reviewer frequency by author/labels).

Machine learning models (logistic regression with features from title, labels, body, and spaCy-extracted keyphrases).

End-to-end pipeline: from fetching PRs → cleaning → training → evaluation → UI to interact with the system.

## Repository Structure
.
├── notebooks/                # Jupyter notebooks
│   ├── data_load_and_clean.ipynb
│   ├── exploratory_analysis.ipynb
│   └── modelling_and_recommendations.ipynb
├── src/                      # Core Python utilities
│   ├── data_utils.py         # Cleaning PR data, body/keyphrases extraction
│   ├── model_utils.py        # Heuristic + ML models and evaluation
│   └── ingest/pr_fetcher/    # API code to fetch PRs from GitHub
├── ui/                       # React frontend - future development
├── data/
│   ├── raw/                  # Raw PR data (ignored in git)
│   └── clean/                # Clean parquet/csv datasets (ignored in git)
└── README.md


1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

2. Python environment

Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt
python -m spacy download en_core_web_sm
