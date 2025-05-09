# 🏀 NBA Win Prediction App (ARVI Model)

This project uses advanced NBA player stats to predict how many games each team will win in a given season. It does this by calculating a custom metric called **ARVI (Advanced Relative Value Index)** and using it to train a machine learning model.

---

## 🚀 How to Use This App

### 1. **Prepare Your Data**
You need two CSV files:

#### `player_stats.csv`
Get your player data from [Basketball-Reference](https://www.basketball-reference.com/leagues/NBA_2024_totals.html):

- Click **"Share & Export" → "Get table as CSV"**
- Make sure your file has these columns:
  - `Player`, `Team`, `VORP`, `MP`, `G`, `USG%`, `BPM`, `TS%`
- Save as `player_stats.csv`

#### `team_wins.csv`
Create this manually with 2 columns:

| Team | Wins |
|------|------|
| BOS  | 64   |
| DEN  | 57   |
| ...  | ...  |

Use [ESPN](https://www.espn.com/nba/standings), [Basketball Reference](https://www.basketball-reference.com/), or any trusted source to fill in the latest win totals.

---

### 2. **Run the App (Locally)**

#### 🖥 Backend (FastAPI)
```bash
uvicorn nba_arvi_api:app --reload
```

#### 🌐 Frontend (HTML Form)
```bash
python3 -m http.server 8080
```
Then open in your browser:
```
http://localhost:8080/index.html
```

1. Upload your `player_stats.csv`
2. Upload your `team_wins.csv`
3. Click **Run ARVI Model**

✅ The backend will compute team predictions and attempt to display:
- A scatter plot of predicted vs actual wins
- A tool to query any team's prediction and their 5 most valuable players by ARVI

> ⚠️ **Note:** The backend model is fully functional, but the app interface is still a work in progress. Some features like chart rendering and prediction lookup may require additional setup or refinement.

---

## 📦 What's Included
- `nba_arvi_win_model.py` – Python script with the ARVI-based prediction model
- `nba_arvi_api.py` – FastAPI backend that runs the model
- `index.html` – Simple HTML interface to upload your files and query predictions

---

## 📁 Output
You’ll get a file called `team_win_predictions.csv` with:
- Actual and predicted wins
- Error (difference between actual and predicted)

---

## 🛠 Requirements
Install dependencies with:
```bash
pip install fastapi uvicorn pandas xgboost scikit-learn matplotlib seaborn
```

For macOS users, also run:
```bash
brew install libomp
```

---

## 🤝 Credit
Player data comes from [Basketball-Reference.com](https://www.basketball-reference.com), which provides excellent NBA stat coverage.

---

Enjoy building predictions with NBA data! 🏀📊
