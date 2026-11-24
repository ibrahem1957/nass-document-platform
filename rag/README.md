# rag/ (reserved; not implemented)

This folder is reserved for the future retrieval-augmented question-answering module. It owns:
- indexing;
- retrieval;
- grounded answer generation;
- answer-time evidence enforcement.

Before writing code, read:
1. `../docs/RAG_HANDOFF.md`: the brief and the first steps.
2. `../docs/RAG_EVIDENCE_POLICY.md`: rules you must enforce, with required tests.
3. `../docprep/docs/RAG_CONSUMER_CONTRACT.md`: the meaning of every safety field.
4. `../docs/KNOWN_LIMITATIONS.md`: what the evidence cannot support.

Constraints:
- **Input:** consume only validated `prepared_document.json` files.
- **UI:** integrate into the existing Studio application.
- **Optional:** Upload, Extract and Prepare must keep working when this module is absent.
