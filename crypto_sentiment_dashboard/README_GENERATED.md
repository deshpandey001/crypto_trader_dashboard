# Crypto Sentiment Dashboard

A Streamlit app and analysis toolkit for exploring crypto sentiment and building simple ML models to predict trading profitability and trader risk.

## Contents
- `app/` — Streamlit application and UI assets.
- `data/` — Raw CSVs used by the project (fear/greed index, historical data).
- `notebooks/` — Exploratory notebooks with preprocessing and analysis.
- `src/` — Project source code: preprocessing, feature engineering, ML models, visualization, reports.
- `outputs/` — Generated charts, cleaned data, saved models, and reports.
- `assets/` — Static assets and custom CSS.

## Features
- Loads and preprocesses crypto price and sentiment data.
- Visual EDA via Streamlit and Plotly/Matplotlib visualizations.
- Trainable ML pipeline (sampling/fast mode available) with progress streaming to the UI.
- Notebook-driven analysis and report generation.

## Quick Start
Prerequisites: Python 3.10+ and a virtual environment.

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app/streamlit_app.py
```

4. Open the app in your browser at the URL Streamlit prints (usually `http://localhost:8501`).

## Notebooks
Open `notebooks/exploratory_analysis.ipynb` to reproduce preprocessing steps, feature engineering, and model experiments. The notebook contains guarded parsing of timestamp columns and robust preprocessing to prevent KeyError issues.

## ML Models
- Models are implemented in `src/ml_models.py` and can be trained from the Streamlit UI.
- For faster iteration, use the app's "Fast" training mode which trains on a sample of the data.
- Trained models are saved to `outputs/models/`.

## Project Structure (short)
```
crypto_sentiment_dashboard/
  app/
  data/
  notebooks/
  outputs/
  src/
  assets/
  requirements.txt
```

## Notes & Best Practices
- Do not commit virtual environments — add `venv/` to `.gitignore` (a `.gitignore` file is suggested in the repo).
- If you see LF/CRLF warnings on Windows, consider configuring Git's `core.autocrlf` or adding a `.gitattributes` and running `git add --renormalize .`.

## Contributing
- Create a branch and open a PR with tests or a short description of the change.
- Keep changes focused and add a short entry in `README` or a `CHANGELOG` for larger features.

## Contact
For questions about the project structure or running experiments, open an issue or contact the maintainer.
