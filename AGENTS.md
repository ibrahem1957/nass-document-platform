# AGENTS.md

All coding agents (Claude Code, Codex, Cursor, Copilot and others) follow **`CLAUDE.md`** in this
directory. It is the single source of agent instructions. This file intentionally contains no
separate rules, so the two can never contradict each other.

Summary, which is non-authoritative; read `CLAUDE.md`:

- **What this is:** a document extraction (Nass) and preparation (DocPrep) platform with one UI
  (Studio). RAG is future work in `rag/`.
- **Contracts:** the retrieval-safety contract is `docprep/docs/RAG_CONSUMER_CONTRACT.md`; the
  answer-time rules are in `docs/RAG_EVIDENCE_POLICY.md`.
- **Never** fabricate numbers, cells, coordinates, captions or answers. Never rewrite raw OCR. Never
  bypass safety metadata.
- **Never** start RunPod or Qari, run OCR, download models or commit secrets without the owner's
  authorization.
- **Run** `.\scripts\run_tests.ps1` before reporting success. The known failures are listed in
  `docs/TESTING.md`.
