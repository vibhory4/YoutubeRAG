# Next Feature Suggestions

1. Source-grounded answer mode with citations
- Add chunk-level citations (`video_title`, `timestamp` when available, `video_url`) to every agent answer.
- Return a structured response object from `ChannelAgent.chat()` that separates `answer`, `citations`, and `confidence`.

2. Evaluation harness for RAG quality and persona fidelity
- Add an offline eval command (for example `python main.py eval @channel`) with a gold QA set per channel.
- Track metrics like retrieval hit-rate@k, groundedness, hallucination rate, and persona consistency over time.
