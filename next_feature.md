# Next Feature Suggestions

1. First-Run Chat Readiness Guardrail
- Add `python main.py bootstrap @channel --limit 3` that runs index + persona + state setup in one command and confirms the channel appears in Chainlit.
- In `app.py`, replace "No channels indexed yet" with a one-click action that triggers indexing for a pasted channel URL/handle.

2. Source-Cited Chat Answers in Chainlit
- Make `ChannelAgent` append compact source citations (`video title + URL`) for each answer so users can verify claims quickly.
- Expose the exact retrieval queries/chunk count in a cleaner UI step (keep current tool debug, but add a user-facing citation block under each response).
