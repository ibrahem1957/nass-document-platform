"""Phase E2-A: local OCR/VLM evaluation on rendered page images.

Models receive only a rendered page image and a static prompt template. Ground
truth and extractor text are never passed to inference. Remote inference is
disabled in E2-A.
"""

# Hard switch: no E2-A code path may send a document or image off this machine.
REMOTE_INFERENCE_ENABLED = False
