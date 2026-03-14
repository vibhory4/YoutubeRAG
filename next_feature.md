# Next Feature Suggestions

1. Bootstrap + In-App Indexing Flow
- Add `python main.py bootstrap @channel --limit N` that runs index + persona + session readiness in one command and prints post-check stats.
- In `app.py`, when no channels exist, use `AskUserMessage` to capture a channel handle and trigger indexing from inside Chainlit (with progress updates).

2. Structured Grounded Answer Contract
- Force a strict response template in `ChannelAgent`: `Answer`, `Evidence`, `Sources`, `Confidence` for every factual query.
- Add an automated eval command (`main.py eval-grounding @channel`) that checks citation presence + claim/source overlap on sampled prompts.
