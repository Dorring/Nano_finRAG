# NF-V2-03 R1B Formal Attempt 5

The run stopped at the first benchmark request because the provider returned a
structured JSON object whose `BOUND` slot contained an invalid number of fact
handles. The constrained DTO rejected it before adapting to the frozen
EvidenceBinding contract. No prediction artifact or Gold scoring was created;
there was no retry and no semantic metric was computed.
