# Validate

Run full validation on the paper trading script:

1. Run `python3 -m py_compile paper_trading_v20_7.py`
2. Run `python3 -m py_compile features.py`
3. Run `python3 -m py_compile config.py`
4. Count lines: `wc -l paper_trading_v20_7.py`
5. Check for syntax issues: `python3 -c "import ast; ast.parse(open('paper_trading_v20_7.py').read()); print('AST OK')"`
6. Verify feature count: `grep -c "'" paper_trading_v20_7.py | head -5` and manually confirm 23 features in raw_features list
7. Report: PASS/FAIL for each check
