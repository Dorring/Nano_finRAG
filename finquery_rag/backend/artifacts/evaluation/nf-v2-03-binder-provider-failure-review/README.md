NF-V2-03 Binder Provider Failure Review

This audit did not rerun the 72-question benchmark and did not read Gold. Synthetic packets used the frozen Binder prompt and EvidenceBinding schema. The current flash-model stress run returned HTTP 2xx with valid JSON in an alternate schema shape, so the gate stops without parser relaxation, prompt changes, model changes, retries, or formal replay.
