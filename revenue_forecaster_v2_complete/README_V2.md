# Revenue Forecasting Simulator V2

## Deploy to Streamlit Community Cloud

1. Replace `revenue_forecaster.py` in the repository with `revenue_forecaster_v2.py` and rename it to `revenue_forecaster.py`.
2. Replace `requirements.txt` with `requirements_v2.txt` and rename it to `requirements.txt`.
3. Commit and push to the branch used by Streamlit Cloud.

```bash
cp revenue_forecaster_v2.py revenue_forecaster.py
cp requirements_v2.txt requirements.txt
git add revenue_forecaster.py requirements.txt
git commit -m "Upgrade revenue forecaster to V2"
git push
```

## V2 features

- SQLite save/load/delete forecast library
- Portable scenario JSON export
- Stage-based win probabilities with manual override
- Capture manager, prime/sub, and prime organization inputs
- Monte Carlo monthly and fiscal-year forecasts
- Executive summary, agency and manager views
- Opportunity sensitivity ranking
- CSV and executive PDF export

## Important hosting note

Streamlit Community Cloud local files can be ephemeral. Treat downloaded JSON scenarios as the durable portable save format. A managed database should be used for durable multi-user storage.
