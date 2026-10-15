# Fake News Detection Using NLP
BCA internship project starter based on the supplied project synopsis.

## What this project does
- Accepts a pasted headline or article in a Streamlit web interface.
- Converts text into TF-IDF features.
- Trains a Logistic Regression text classifier.
- Evaluates it on a held-out test split using accuracy, precision, recall, F1-score and a confusion matrix.
- Saves the fitted pipeline for use by the application.

**Important:** this is a text-classification prototype, not a fact-checking service. Predictions depend on the dataset and may be wrong.

## Project structure
```text
fake_news_nlp_internship/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── news.csv
└── models/
```

## Setup in VS Code on Windows
Open this project folder in VS Code. In the terminal, run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use the environment's Python directly:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use a Python version supported by the installed scikit-learn release. If your Python installation is too new for a package, install a supported Python version and recreate the virtual environment.

## Dataset setup
The included `data/news.csv` is a **tiny synthetic DEMO dataset only** to show the expected format. Its scores are not meaningful and must not be presented as real project results.

For the actual internship project, replace it with a documented, labelled dataset that you are permitted to use. Keep a record of its title, creator/organisation, source URL, licence, access date, label definitions and known limitations.

Required CSV format:
```csv
text,label
"Article text goes here",real
"Another article goes here",fake
```

The training script also recognises common alternatives such as `title`, `content`, `article`, `class`, `target`, and `category`. If a dataset has separate title and article columns, combine them into a single text field or adjust the script.

## Train the model
After placing your real dataset at `data/news.csv`, run:
```powershell
python train_model.py
```

The script creates:
- `models/fake_news_pipeline.joblib`
- `models/evaluation_metrics.json`

Review the measured metrics and confusion matrix. Do not claim an accuracy score before running the experiment. Check for duplicates and related stories across train/test splits; leakage can inflate results.

## Launch the website
```powershell
streamlit run app.py
```
Streamlit will show a local URL in the terminal, commonly `http://localhost:8501`.

## Suggested internship demonstration
1. Explain the problem statement and objectives.
2. Show the dataset and explain the labels and licence.
3. Demonstrate preprocessing and TF-IDF.
4. Explain the train/test split and Logistic Regression.
5. Run the app with sample inputs.
6. Present measured accuracy, precision, recall, F1-score and confusion matrix.
7. Explain limitations and why predictions need independent verification.

## Future enhancements
- Compare Logistic Regression with Multinomial Naive Bayes and Linear SVM.
- Add more diverse labelled data and evaluate language/topic differences.
- Add a link-based fact-checking workflow, while avoiding claims that text style alone proves truth.
- Add careful uncertainty calibration and a clearer user explanation.
