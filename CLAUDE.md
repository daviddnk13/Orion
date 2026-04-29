# CLAUDE.md — Orion Crypto Trading System

## 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

- State your assumptions explicitly. If uncertain, ask.
- If a function exists in another script in this repo, READ IT FIRST before reimplementing.
- When working with external APIs (CCXT, OKX, Binance), verify limits and pagination behavior.
- If your implementation produces different numbers than expected (e.g., edge_rate 82% when spec says 18%), STOP and report the discrepancy. Do NOT proceed.

ORION-SPECIFIC:
- orion_crypto.py is PRODUCTION (Track A). NEVER modify it.
- lab/ scripts are EXPERIMENTAL (Track B). All new work goes here.
- When a spec says "copy function X from script Y", COPY IT EXACTLY. Do not rewrite.

## 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- If a working function exists in the codebase, reuse it — don't reimagine it.
- If told to use h=12, k=1.0, d=0.5: use exactly those values. Don't add flexibility.

ORION-SPECIFIC:
- compute_target() is a validated, tested function. Copy it verbatim when needed.
- fetch_all_ohlcv() with limit=300 is proven to work. Don't change the limit.
- Don't add logging, monitoring, or error handling beyond what's requested.

## 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

- When fixing a bug, change ONLY the lines that fix it.
- Don't "improve" adjacent code, comments, or formatting.
- When told to fix 3 instances of a pattern, find and fix ALL of them. Verify count.
- Match existing style: if the codebase uses single quotes, use single quotes.

ORION-SPECIFIC:
- When fixing a variable reference (e.g., sharpe_edge0 → r['sharpe_edge0']),
  grep the ENTIRE file for similar patterns. Fix ALL instances.
- Format strings: if told to change ":+.1pp" to "*100:+.1f", grep for ALL occurrences.
  Report the count found vs count fixed.

## 4. Goal-Driven Execution
Define success criteria. Loop until verified.

Every task must have verification:
1. [Step] → verify: [check]
2. [Step] → verify: [check]

ORION-SPECIFIC verification checklist:
- After implementing compute_target(): verify edge_rate is 15-25% (NOT 80%+)
- After implementing fetch_all_ohlcv(): verify 8000+ bars downloaded (NOT 300)
- After any fix: python3 -m py_compile [file] must pass
- After copying a function: diff the original vs copy, they should be identical
- After fixing N instances of a pattern: grep confirms 0 remaining instances

## 5. Orion Project Rules

### Architecture
- Track A (Production): ~/orion/orion_crypto.py — NEVER MODIFY
- Track B (Lab): ~/orion/lab/ — all experimental scripts here
- Venv: ~/orion/venv/ — always activate before running

### Common Pitfalls (LEARNED FROM EXPERIENCE)
1. OKX CCXT pagination: use limit=300, NOT 1000. OKX returns max ~300 per request.
2. compute_target(): edge=1 means the condition PASSES (big move + clean path).
   If edge_rate > 50%, the logic is INVERTED — stop and fix.
3. Format strings: Python has no "pp" format spec. Use f"{value*100:+.1f}pp"
4. Variable scope in loops: when building result dicts, use r['key'] not bare variable names.
5. Telegram: BOT_TOKEN from env var, never hardcoded. Chat ID: -1003505760554, Topic: 972.
6. Dependencies: ccxt, scipy, scikit-learn, lightgbm must be in venv.

### Self-Check Before Reporting "Done"
- [ ] py_compile passes
- [ ] All instances of a pattern are fixed (grep to confirm)
- [ ] Numerical outputs match expected ranges from spec
- [ ] No functions were reimplemented when copy was specified
- [ ] Track A (orion_crypto.py) was NOT modified
