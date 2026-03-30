# 🐟 Fish Weight Prediction App

Full assessment covering EDA, Feature Engineering, Model Building and Evaluation.

## Files
| File | Description |
|------|-------------|
| `app.py` | Streamlit application (all 4 parts + predictor) |
| `Fish_Assessment.ipynb` | Jupyter notebook with all answers |
| `Linear__1_assement.csv` | Dataset (476 fish, 7 species) |
| `fish_model.pkl` | Trained LinearRegression model |
| `model_columns.pkl` | Feature column list for inference |
| `requirements.txt` | Python dependencies |

## Deploy on Streamlit Cloud

1. Push this folder to a **GitHub repository** (all files at repo root)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set **Main file path** to `app.py`
4. Click **Deploy** ✅

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
