"""Optional, model-free semantic verification: does the text after a candidate heading look like it develops
the heading's topic?

This is supporting evidence only. Shared words are never proof, and their absence is never a rejection.
A local semantic model can later be added by implementing the SemanticVerifier protocol and registering it
in `make_verifier`, without changing the rest of the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .text_patterns import ends_with_colon, ends_with_terminal, match_text, words

LIMITATIONS = ("Lexical and structural cues only (no language model, no embeddings). Shared terms and their "
               "absence are both inconclusive; the result can only support or weaken other evidence.")

AR_STOP = {"في", "من", "على", "الى", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "التي", "الذي", "الذين", "او",
           "أو", "ثم", "كما", "قد", "لقد", "كان", "كانت", "هو", "هي", "هم", "ما", "لا", "لم", "لن", "ان", "أن", "إن",
           "كل", "بعض", "بين", "حيث", "عند", "عدد", "و", "تم", "يتم", "اي", "أي", "ومن", "وفي", "به", "بها", "لها",
           "له", "منها", "فيها", "عليها", "ابرز", "أبرز", "ايضا", "أيضا", "وفق", "حسب", "خلال", "بعد", "قبل", "لدى",
           "نحو", "ضمن", "دون", "غير", "جميع", "هناك", "هنا", "ذات", "عدة", "مدى", "مثل", "وهو", "وهي",
           "اللذين", "التى", "إلي", "عليه", "عنه", "منه", "فيه", "يكون", "تكون", "ليس", "ليست",
           "اذا", "إذا", "لكن", "بل", "حتى", "منذ", "وقد", "ولا", "وما", "فقد", "انه", "أنه", "انها", "أنها"}
EN_STOP = {"the", "a", "an", "of", "in", "on", "and", "or", "to", "for", "with", "by", "is", "are", "was", "were",
           "this", "that", "these", "those", "as", "at", "be", "it", "its", "from", "which", "their", "into",
           "been", "being", "has", "have", "had", "will", "would", "can", "could", "should", "may", "might", "not",
           "but", "if", "than", "then", "so", "such", "more", "most", "other", "about", "over", "after", "before",
           "between", "during", "each", "all", "any", "both", "per", "via", "our", "them", "they", "we", "you",
           "your", "his", "her", "he", "she", "who", "whom", "what", "when", "where", "why", "how", "also", "there"}
# Prepositions ب/ل/ك are stripped only together with the article (بال، لل، كال): on their own they are very often
# the first root letter (بيانات، لجنة، كتاب). Only the conjunctions و/ف are stripped alone.
AR_PREFIXES = ("وبال", "وال", "بال", "كال", "فال", "لل", "ال", "و", "ف")
AR_SUFFIXES = ("ات", "ون", "ين", "ان", "ية", "ها", "هم", "ه", "ة", "ي")
EXPLANATORY_START = ("تم ", "يتم ", "يتناول", "يوضح", "توضح", "يعرض", "تعرض", "يشمل", "تشمل", "يهدف", "تهدف", "يستعرض",
                     "يبين", "تبين", "this section", "this chapter", "we ", "the following", "here ", "includes",
                     "describes", "presents", "shows")
CONTINUATION_START = ("و ", "ثم ", "كما ", "لكن ", "إلا ", "الا ", "and ", "but ", "however", "also ", "moreover",
                      "furthermore")


@dataclass
class SemanticEvidence:
    verifier: str
    score: float                       # -1 .. 1
    supporting: list[str] = field(default_factory=list)
    conflicting: list[str] = field(default_factory=list)
    limitations: str = LIMITATIONS

    def to_dict(self) -> dict:
        data = asdict(self)
        data["score"] = round(self.score, 3)
        return data


class SemanticVerifier(Protocol):
    name: str

    def verify(self, heading: str, following: list[str]) -> SemanticEvidence: ...


class NullVerifier:
    name = "none"

    def verify(self, heading: str, following: list[str]) -> SemanticEvidence:
        return SemanticEvidence(self.name, 0.0, [], [], "semantic verification disabled")


def light_stem(token: str) -> str:
    t = match_text(token).lower()
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    if re.search(r"[ء-ي]", t):
        for prefix in AR_PREFIXES:
            if t.startswith(prefix) and len(t) - len(prefix) >= 3:
                t = t[len(prefix):]
                break
        for suffix in AR_SUFFIXES:
            if t.endswith(suffix) and len(t) - len(suffix) >= 3:
                t = t[:-len(suffix)]
                break
        return t
    for suffix in ("ing", "ed", "es", "s"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            return t[:-len(suffix)]
    return t


def content_terms(text: str) -> set[str]:
    out = set()
    for token in words(text):
        low = token.lower()
        if low in AR_STOP or low in EN_STOP or token.isdigit() or len(token) < 3:
            continue
        out.add(light_stem(token))
    return out


class HeuristicVerifier:
    name = "heuristic"

    def verify(self, heading: str, following: list[str]) -> SemanticEvidence:
        evidence = SemanticEvidence(self.name, 0.0)
        following = [f for f in following if f and f.strip()]
        if not following:
            evidence.conflicting.append("no following content before the next candidate or the document end")
            evidence.score = -0.5
            return evidence
        score = 0.0
        first = " ".join(following[0].split())
        first_low = match_text(first).lower()
        heading_terms = content_terms(heading)
        following_terms = content_terms(" ".join(following))
        shared = sorted(heading_terms & following_terms)
        if shared and heading_terms:
            part = 0.3 * min(1.0, len(shared) / max(1, len(heading_terms)))
            score += part
            evidence.supporting.append(f"shares {len(shared)} content term(s) with the following text "
                                       f"({', '.join(shared[:6])}); repetition alone is not proof")
        elif heading_terms:
            evidence.supporting.append("no shared content terms (not counted against the heading)")
        if ends_with_colon(heading):
            score += 0.2
            evidence.supporting.append("ends with a colon that introduces the following content")
        if len(words(first)) >= 2 * max(1, len(words(heading))) and len(words(first)) >= 6:
            score += 0.2
            evidence.supporting.append("followed by a longer explanatory passage")
        if first_low.startswith(EXPLANATORY_START):
            score += 0.1
            evidence.supporting.append("the following text opens like an explanation")
        if first_low.startswith(CONTINUATION_START):
            score -= 0.2
            evidence.conflicting.append("the following text opens like the continuation of an earlier sentence")
        if ends_with_terminal(heading) and len(words(heading)) > 6:
            score -= 0.2
            evidence.conflicting.append("the candidate itself reads as a complete sentence")
        if len(words(first)) < 3:
            score -= 0.2
            evidence.conflicting.append("the following text is very short")
        evidence.score = max(-1.0, min(1.0, score))
        return evidence


def make_verifier(name: str, enabled: bool = True) -> SemanticVerifier:
    if not enabled or name == "none":
        return NullVerifier()
    if name == "heuristic":
        return HeuristicVerifier()
    raise ValueError(f"unknown semantic verifier: {name}")
