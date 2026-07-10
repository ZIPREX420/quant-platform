# Candidate registry (ADR-0006 - paper trading only)

Every `*.json` file here is an UNVALIDATED strategy definition under forward
test on paper. Contract (enforced by `load_candidate()`, fail-closed):

- validates against the candidate schema (strategy schema minus
  `validation_report`, plus required `tracking`);
- must NOT contain a `validation_report` (validated strategies live in
  `config/strategies/`);
- must carry `tracking.prediction`: a pre-registered, falsifiable statement
  of what the paper record is expected to show and why.

Candidates are hypotheses, not recommendations. They can only ever reach
`PaperTradingSession`; promotion beyond paper requires the full ADR-0005
contract (signed validation report), which this directory cannot satisfy
by construction.
