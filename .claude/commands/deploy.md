# Deploy

Deploy changes to paper trading:

1. Run /validate first — abort if any FAIL
2. `git add -A && git status`
3. Show diff summary: `git diff --cached --stat`
4. Ask for commit message confirmation
5. `git commit -m "<message>"`
6. `git push origin main`
7. `sudo systemctl restart orion.service`
8. Wait 5 seconds, then `sudo systemctl status orion.service`
9. Report: deployed successfully or failed
