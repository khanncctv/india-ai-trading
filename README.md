# India AI Trading Agent — Cloud Dashboard

A Streamlit cloud-ready paper-trading dashboard.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `.streamlit/config.toml`, and `README.md`.
3. Open Streamlit Community Cloud.
4. Select the repository and `app.py` as the main file.
5. Deploy.

No broker API key is required for this paper-trading version.

## Important

This dashboard uses public market-data access through yfinance for demonstration.
It does not place real orders.

Before live trading, replace the data layer with an appropriate broker/exchange data source and add:
- broker authentication
- order-status reconciliation
- slippage/fees/taxes
- position and exposure limits
- kill switch
- audit logs
- compliance checks
- paper-trading validation

Do not treat backtest results as guaranteed future performance.
