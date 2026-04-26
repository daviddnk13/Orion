# Paper Trading Status

Check current paper trading status:

1. `sudo systemctl status orion.service`
2. `sudo journalctl -u orion.service --no-pager -n 30`
3. `cat ~/orion/paper_trading_state.json`
4. Check last CSV log entry: `tail -1 ~/orion/paper_trading_log.csv`
5. Report: service status, last cycle time, current balance, current position, any errors
