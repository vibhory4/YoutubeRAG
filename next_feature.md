# Next Feature Suggestions

1. Bootstrap + First-Run UX
- Add `python main.py bootstrap @channel --limit N` to run indexing, persona creation, and a final health check in one command.
- In `app.py`, let a user submit a channel handle from the empty state and start indexing without leaving the UI.

2. Retrieval Quality Evaluation
- Add `python main.py eval @channel` to score retrieval relevance, empty-result rate, and citation coverage on a saved prompt set.
- Store eval history in `data/evals/` so you can compare retrieval quality after chunking or prompt changes.
