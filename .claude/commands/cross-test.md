# Cross-Asset Test

Run cross-asset generalization test (ETH model on BTC/SOL):

1. Verify model exists: `ls -la ~/orion/model_v20_6_1.pkl`
2. Verify CCXT works: `python3 -c "import ccxt; print(ccxt.okx().fetch_ticker('ETH/USDT')['last'])"`
3. Run: `python3 ~/orion/cross_asset_test.py`
4. Report results table and verdict (GENERALIZA vs NO GENERALIZA)
