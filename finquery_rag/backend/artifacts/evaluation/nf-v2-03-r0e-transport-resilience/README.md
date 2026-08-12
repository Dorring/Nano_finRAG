# NF-V2-03 R0E

The transport policy was sealed with SDK retries disabled, one semantic
response budget, one transport retry budget, and a fixed three-second delay.
The requested repeat stability matrices were explicitly skipped. The prior
`qwen3.7-max-2026-05-17` trial required `enable_thinking=true` and was not
continued. Formal Attempt 3 then ran with `qwen3.7-plus` while the frozen
Binder configuration remained `thinking=false`.

Gold reads: 0. Production remains V1.
