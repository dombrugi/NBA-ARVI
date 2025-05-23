# 📐 ARVI (Advanced Relative Value Index)

**ARVI** is a custom-designed basketball metric developed for this project to estimate a player's relative contribution to team success using both opportunity and efficiency-based measures. It integrates multiple advanced statistics into a single, interpretable index that reflects each player's real-world value in the context of their team.

---

## 📊 Metric Definition

The ARVI metric is calculated as:

```
ARVI = (α · VORP_Share + β · BPM_Share) × (γ · MPG_Share + δ · USG_Share) × η × TS_Norm
```

Where:
- **VORP_Share**: Player's Value Over Replacement (VORP) as a proportion of team total
- **BPM_Share**: Box Plus/Minus normalized to the team average
- **MPG_Share**: Minutes per game normalized to team total
- **USG_Share**: Usage percentage normalized to team total
- **TS_Norm**: True Shooting Percentage normalized to the team average

---

## 🎯 Conceptual Motivation

ARVI is designed to reflect not just raw output, but a player's **scaled contribution** relative to teammates:
- High ARVI values typically indicate players with strong impact (VORP/BPM), large roles (minutes and usage), and above-average efficiency (TS%).
- ARVI rewards players who are both heavily relied on and efficient — but adjusts for context, ensuring that bench players and stars are evaluated fairly within team structure.

---

## 🏀 Practical Application

This metric powers a machine learning model that predicts team win totals across an NBA season. Aggregated team-level ARVI features (sum, mean, max, standard deviation, top-3 sum) were found to be strong predictors of success, with **ARVI standard deviation inversely correlated with win totals**, highlighting the predictive value of **roster balance**.
