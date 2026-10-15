# Revenue Forecasting Simulator

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run revenue_forecaster.py
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

## Inputs

Use the editable table or upload CSV/XLSX with these exact columns:

- Program
- Status (`Booked` or `Bid`)
- Amount
- Start Date
- End Date
- Win Probability (0 to 1)
- Funding Agency

The application allocates each program amount linearly over inclusive calendar months and samples each bid independently during each Monte Carlo run.
