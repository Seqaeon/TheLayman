from __future__ import annotations

import re
from datetime import datetime, timezone

from the_layman.backend.schemas import Explanation, WhyItMatters
from the_layman.pipeline.ingestion import PaperContent
from the_layman.pipeline.llm_client import generate_json_with_debug, model_version_tag

from the_layman.pipeline.llm_client import generate_json_with_debug, model_version_tag

PROMPT_VERSION = "layman-prompt-v3"

_JARGON_MAP = {
    "LLM": "AI text model",
    "LLMs": "AI text models",
    "autonomous": "self-running",
    "hierarchical multi-agent": "team-based",
    "granularity": "how small each task is",
    "institutional investors": "large professional investors",
    "empirical": "based on measured results",
    "recurrent neural networks": "older step-by-step AI models",
    "convolutional neural networks": "pattern-finding AI models",
    "encoder-decoder": "read-then-write setup",
    "sequence transduction": "converting one sequence into another (like one sentence to another)",
    "language modeling": "predicting likely next words",
    "machine translation": "automatic language translation",
}


def _de_jargon(text: str) -> str:
    if not text:
        return "unknown"
    out = text
    for src, dst in _JARGON_MAP.items():
        out = out.replace(src, dst)
    # Collapse excessive whitespace
    out = re.sub(r"\s+", " ", out).strip()
    return out or "unknown"


def _is_unknown(value: str) -> bool:
    if not value:
        return True
    return value.strip().lower() in {"unknown", "n/a", ""}


def build_prompt(paper: PaperContent) -> str:
    return f"""
You are THE LAYMAN — a brilliant science communicator who translates dense academic research
into clear explanations for a smart, curious friend with zero scientific background.

You do NOT summarize like a researcher.
You TRANSLATE complex ideas into everyday understanding.

Use ONLY the provided paper text.
If a detail is missing → output "unknown".
Never guess or fabricate.

---

CORE GOAL:
Turn technical research → simple human understanding.

---

CRITICAL CONTENT & TONE RULES:

1. READING LEVEL
- Write at 9th-grade level or lower.
- Short clear sentences.
- Active voice only.
- Use common everyday words.
- No academic tone.

2. JARGON & METRIC BAN
- Do NOT use scientific terms, acronyms, or metrics unless immediately translated.
- Always explain in plain English.
Example:
"they scored 28.4 on a benchmark" → "they performed better than earlier systems".

3. OLD WAY vs NEW WAY FRAMEWORK (MANDATORY)
In coffee_chat:
- Clearly explain the frustrating or limited "Old Way".
- Then explain the paper's "New Way" as the improvement.

4. MANDATORY ANALOGY (VERY IMPORTANT)
- coffee_chat must revolve around ONE relatable everyday analogy.
- Use physical or familiar experiences (traffic, cooking, maps, tools, recipes, assembly lines, etc.).
- The analogy should explain how the method works.

5. GROUNDED REALITY
- Zero hype or marketing language.
- Use concrete real-world effects.
- If impact is not clearly stated → "unknown".

---

STYLE REQUIREMENTS:

- Sound like explaining to a friend over coffee.
- Conversational but precise.
- Calm, clear, and honest.
- Prefer concrete examples over abstract language.
- Explain unfamiliar ideas step-by-step.

If a sentence sounds like a research paper, rewrite it.

---

STYLE EXAMPLE (follow style, not wording):

Problem:
Older systems processed information step-by-step, which made them slow and inefficient.

Analogy:
Imagine a single checkout line at a store where everyone must wait.

Solution:
A newer approach lets many checkouts run at once, making everything faster.

Impact:
This improves tools people use daily.

Limitations:
It still requires significant resources.

Always explain like this: simple, practical, conversational.

---

INPUT DATA:

<paper_text>
Paper URL: {paper.url}

Abstract:
{paper.abstract}

Introduction:
{paper.introduction}

Conclusion:
{paper.conclusion}
</paper_text>

---

QUALITY CHECK BEFORE OUTPUT:
- Could a high school student understand this?
- Did you remove technical language?
- Did you explain the Old Way vs New Way clearly?
- Did you use one strong analogy?
- Did you explain real-world impact?
If not, rewrite to be simpler.

---

OUTPUT FORMAT:
Return STRICTLY valid JSON matching this exact structure. No markdown formatting, no extra text before or after the JSON.

CRITICAL PARAGRAPH RULE — READ CAREFULLY:
For coffee_chat and deep_dive you MUST separate each paragraph using the two-character sequence \\n\\n (a backslash, letter n, backslash, letter n) inside the JSON string.
Do NOT run paragraphs together as one long string.
Example of WRONG output (paragraphs run together):
  "coffee_chat": "Para 1 text. Para 2 text. Para 3 text."
Example of CORRECT output (paragraphs separated by \\n\\n):
  "coffee_chat": "Para 1 text.\\n\\nPara 2 text.\\n\\nPara 3 text."

{{
  "reasoning_scratchpad": "Write out your 7-step reasoning process here first: 1. main idea 2. problem 3. concept translation 4. step-by-step solution 5. everyday connection 6. real-world impact 7. limitations.",
  "core_claim": "One single clear sentence. No jargon.",
  "twitter_summary": "2–3 sentences. Hook the reader. State the finding. Explain why normal people should care.",
  "coffee_chat": "Write EXACTLY 4 paragraphs, each separated by \\n\\n. Do NOT merge them into one block. Paragraph 1: The Problem — explain the Old Way and why it was slow or limited (4–5 sentences). Paragraph 2: The Analogy — one relatable everyday comparison that explains the method (4–5 sentences). Paragraph 3: The Solution — how the New Way fixes the problem (4–5 sentences). Paragraph 4: Real-world impact — what this means for real people today (3–4 sentences).",
  "deep_dive": "Write EXACTLY 5 paragraphs, each separated by \\n\\n. Do NOT merge them. Start each paragraph with its label in the format 'Label: rest of text'. Each paragraph must be 6–8 sentences. 9th-grade reading level. Provide a much deeper, more explanatory exploration of each topic than usual. The 5 required sections:\n1. Context: What existed before? Why was it a problem? Who cared and why?\n2. How It Works: Extensive step-by-step explanation of the new method. Use a concrete real-world example. Explain each step plainly.\n3. Evidence: What experiments were run? What did results show — in plain terms, not numbers? How confident can we be?\n4. Assumptions: What must be true for this to work well? What does it rely on? What could break it?\n5. Limitations: What does this NOT fix? What new problems does it create? What is still unknown?",
  "why_it_matters": {{
    "who_it_affects": "specific people or industries if supported, else 'unknown'",
    "problems_solved": "concrete real-world problems",
    "timeline_of_impact": "realistic timeline or 'unknown'",
    "limitations": "practical constraints or risks"
  }},
  "confidence_level": "high / medium / low based on clarity of evidence."
}}
""".strip()


def _coerce_llm_output(raw: dict, paper: PaperContent) -> Explanation:
    """Map raw LLM JSON dict → Explanation schema, applying jargon cleanup."""
    why_raw = raw.get("why_it_matters") or {}

    # Banned hype words — replace with "suggests"
    _BANNED = ["breakthrough", "revolutionary", "proves", "guarantees"]

    def _clean(text: str) -> str:
        if not text or _is_unknown(text):
            return "unknown"
        out = text
        for word in _BANNED:
            out = out.replace(word, "suggests")
        return _de_jargon(out)

    return Explanation(
        core_claim=_clean(raw.get("core_claim") or "unknown"),
        twitter_summary=_clean(raw.get("twitter_summary") or "unknown"),
        coffee_chat=_clean(raw.get("coffee_chat") or "unknown"),
        deep_dive=_clean(raw.get("deep_dive") or "unknown"),
        why_it_matters=WhyItMatters(
            who_it_affects=_clean(why_raw.get("who_it_affects") or "unknown"),
            problems_solved=_clean(why_raw.get("problems_solved") or "unknown"),
            timeline_of_impact=_clean(why_raw.get("timeline_of_impact") or "unknown"),
            limitations=_clean(why_raw.get("limitations") or "unknown"),
        ),
        confidence_level=raw.get("confidence_level") or "unknown",
        original_paper_link=paper.url or "unknown",
        sources_used=[paper.source, "generated"],
        generated_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _failed_explanation(paper: PaperContent) -> Explanation:
    """Return an honest all-unknown Explanation when the model fails."""
    return Explanation(
        core_claim="unknown",
        twitter_summary="unknown",
        coffee_chat="unknown",
        deep_dive="unknown",
        why_it_matters=WhyItMatters(
            who_it_affects="unknown",
            problems_solved="unknown",
            timeline_of_impact="unknown",
            limitations="unknown",
        ),
        confidence_level="low",
        original_paper_link=paper.url or "unknown",
        sources_used=[paper.source, "failed"],
        generated_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def build_explanation_with_debug(paper: PaperContent, user_id: str = "default") -> tuple[Explanation, dict | None, str, str | None]:
    prompt = build_prompt(paper)
    llm_result, llm_raw_text = generate_json_with_debug(prompt, user_id=user_id)

    if llm_result and isinstance(llm_result, dict):
        explanation = _coerce_llm_output(llm_result, paper)
        return explanation, llm_result, prompt, llm_raw_text

    # Model failed — return honest failure, don't inject hardcoded content
    return _failed_explanation(paper), llm_result, prompt, llm_raw_text


def build_explanation(paper: PaperContent, user_id: str = "default") -> Explanation:
    explanation, _, _, _ = build_explanation_with_debug(paper, user_id=user_id)
    return explanation
