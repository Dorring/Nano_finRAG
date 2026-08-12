# NF-V2-01 R1 Transport Isolation

The first matrix isolated a CRLF-tainted Authorization header. The runner now strips surrounding environment transport whitespace; post-fix checks are synthetic-only and sequential.
