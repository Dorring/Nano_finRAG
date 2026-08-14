# NF-V2-09 R2.2.1 Final SFT Export Preflight

This artifact is a structural projection of the canonical R2.2.1 dataset.
Each output JSONL line is exactly the parsed `messages` array from one
canonical row; no semantic field or message content is changed.

- Canonical rows: 2100
- Canonical SHA256: `4771803fd179bdbb9bf146dd7be782fe7a9de40d83408d548d181e5f1e233753`
- Training rows: 2100
- Training SHA256: `299290d9ace0018336acabc12a2e90f5d9c67f1ba4beadbb8673f904a59c8390`
- Message hash matches: 2100/2100
- Loader: `tasks/customjson.py`
- Context limit checked: 2048
- Training/model/retrieval calls: 0/0/0

The tokenizer smoke test uses `nanochat.tokenizer.get_tokenizer()` and the
same `render_conversation` path used by `scripts/chat_sft.py`. User content is
masked out and assistant content is supervised; no optimizer or model forward
pass is run.
