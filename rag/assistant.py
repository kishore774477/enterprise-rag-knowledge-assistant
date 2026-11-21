"""The RAG assistant: retrieve relevant chunks, then answer with citations.

* OpenAI mode: a LangChain LCEL chain (prompt | ChatOpenAI | parser) answers from context.
* Offline mode (or if the OpenAI call fails): an extractive answerer picks the most
  relevant sentences from the retrieved chunks, so answers are always grounded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import Settings
from .embeddings import HashingTfidfEmbeddings
from .knowledge_base import KnowledgeBase, SearchResult
from .text import split_sentences, tokenize

logger = logging.getLogger(__name__)

NOT_FOUND = "I couldn't find this in the knowledge base. Please contact the relevant department (HR, Finance or IT) for help."

SYSTEM_PROMPT = """You are an enterprise knowledge assistant for company employees.
Answer the question using ONLY the numbered context passages below, which come from
official company documents (HR policies, finance manuals, IT policies, FAQs).

Rules:
- Be concise and specific: include exact numbers, limits, deadlines and approvers.
- Cite the passages you used inline, like [1] or [2][3].
- If the context does not contain the answer, reply exactly: "{not_found}"
- Never invent policy details that are not in the context.

Context:
{context}"""

CONDENSE_PROMPT = """Given the conversation so far and a follow-up question, rewrite the
follow-up as a standalone question that can be understood without the conversation.
Return only the rewritten question.

Follow-up question: {question}"""


@dataclass
class Source:
    index: int
    source: str
    location: str
    score: float
    content: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    mode: str = "offline"
    note: str = ""

    @property
    def found(self) -> bool:
        return self.answer != NOT_FOUND


def _to_messages(history: list[dict] | None, max_turns: int = 3) -> list:
    messages = []
    for msg in (history or [])[-2 * max_turns :]:
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if msg.get("role") == "user":
            messages.append(HumanMessage(content))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content.split("\n\n**Sources**")[0]))
    return messages


class RAGAssistant:
    def __init__(self, settings: Settings, knowledge_base: KnowledgeBase | None = None):
        self.settings = settings
        self.kb = knowledge_base or KnowledgeBase(settings)
        self._llm = None
        self.fallback_reason = ""

    # -- public API ----------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self.settings.mode

    def prepare(self) -> dict:
        """Load or build the index. If OpenAI is unreachable, switch to offline mode."""
        try:
            return self.kb.ensure_ready()
        except Exception as exc:
            if not self.settings.use_openai:
                raise
            self._fallback_to_offline(exc)
            return self.kb.ensure_ready()

    def _fallback_to_offline(self, exc: Exception) -> None:
        logger.warning("OpenAI unavailable (%s: %s); switching to offline mode", type(exc).__name__, exc)
        self.settings = self.settings.with_overrides(requested_mode="offline")
        self.kb = KnowledgeBase(self.settings)
        self.fallback_reason = (
            f"OpenAI is unavailable ({type(exc).__name__}) - check OPENAI_API_KEY / network. "
            "Running in offline mode."
        )

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        question = (question or "").strip()
        if not question:
            return Answer(question, "Please type a question.", mode=self.mode)

        self.prepare()
        try:
            query = self._standalone_question(question, history)
            results = self.kb.search(query)
        except Exception as exc:
            if not self.settings.use_openai:
                raise
            self._fallback_to_offline(exc)
            return self.ask(question, history)

        if self.fallback_reason:
            answer = self._answer(question, query, results, history)
            answer.note = answer.note or self.fallback_reason
            return answer
        return self._answer(question, query, results, history)

    def _answer(self, question: str, query: str, results: list[SearchResult], history: list[dict] | None) -> Answer:
        sources = [
            Source(i, r.source, r.location, r.score, r.document.page_content)
            for i, r in enumerate(results, start=1)
        ]

        if self.settings.use_openai:
            try:
                text = self._generate_llm(question, results, history)
                return Answer(question, text, sources if text != NOT_FOUND else [], mode="openai")
            except Exception as exc:  # network/auth/quota errors -> stay useful
                logger.warning("OpenAI generation failed, using extractive fallback: %s", exc)
                answer = self._generate_extractive(question, query, results, sources)
                answer.mode = "openai (fallback)"
                answer.note = f"OpenAI request failed ({type(exc).__name__}); showing an extractive answer instead."
                return answer

        return self._generate_extractive(question, query, results, sources)

    # -- query handling ----------------------------------------------------------------------
    def _standalone_question(self, question: str, history: list[dict] | None) -> str:
        prior_user = [m["content"] for m in (history or []) if m.get("role") == "user" and isinstance(m.get("content"), str)]
        if not prior_user:
            return question
        if self.settings.use_openai:
            try:
                prompt = ChatPromptTemplate.from_messages(
                    [MessagesPlaceholder("history"), ("human", CONDENSE_PROMPT)]
                )
                chain = prompt | self._get_llm() | StrOutputParser()
                return chain.invoke({"history": _to_messages(history), "question": question}).strip() or question
            except Exception as exc:
                logger.warning("Could not condense follow-up question: %s", exc)
        # Offline heuristic: short follow-ups ("what about interns?") borrow the previous question.
        if len(tokenize(question)) <= 4:
            return f"{prior_user[-1]} {question}"
        return question

    # -- generation -----------------------------------------------------------------------------
    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self.settings.chat_model,
                temperature=self.settings.temperature,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                timeout=60,
                max_retries=2,
            )
        return self._llm

    def _generate_llm(self, question: str, results: list[SearchResult], history: list[dict] | None) -> str:
        if not results:
            return NOT_FOUND
        context = "\n\n".join(
            f"[{i}] (source: {r.source}{', ' + r.location if r.location else ''})\n{r.document.page_content}"
            for i, r in enumerate(results, start=1)
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), MessagesPlaceholder("history"), ("human", "{question}")]
        )
        chain = prompt | self._get_llm() | StrOutputParser()
        return chain.invoke(
            {"context": context, "not_found": NOT_FOUND, "history": _to_messages(history), "question": question}
        ).strip()

    def _generate_extractive(self, question: str, query: str, results: list[SearchResult], sources: list[Source]) -> Answer:
        relevant = [r for r in results if r.score >= self.settings.min_relevance]
        if not relevant:
            return Answer(question, NOT_FOUND, [], mode="offline")

        embeddings = self.kb.embeddings
        weight = embeddings.term_weight if isinstance(embeddings, HashingTfidfEmbeddings) else (lambda _t: 1.0)
        query_terms = set(tokenize(query))
        total = sum(weight(t) for t in query_terms) or 1.0

        candidates = []  # (score, chunk_rank, position, sentence)
        for rank, result in enumerate(relevant, start=1):
            section = result.document.metadata.get("section", "")
            for pos, sentence in enumerate(split_sentences(result.document.page_content)):
                if sentence == section or len(sentence) < 15:
                    continue  # skip the heading line we prepended, and fragments
                overlap = query_terms & set(tokenize(sentence))
                if not overlap:
                    continue
                score = sum(weight(t) for t in overlap) / total + 0.05 * result.score - 0.02 * rank
                candidates.append((score, rank, pos, sentence))

        if not candidates:
            return Answer(question, NOT_FOUND, [], mode="offline")

        candidates.sort(reverse=True)
        best = candidates[0][0]
        chosen = [c for c in candidates[:3] if c[0] >= 0.5 * best]
        chosen.sort(key=lambda c: (c[1], c[2]))  # keep document order for readability
        lines = [f"- {sentence} [{rank}]" for _score, rank, _pos, sentence in chosen]
        text = "Here is what the company documents say:\n" + "\n".join(lines)
        used = {c[1] for c in chosen}
        return Answer(question, text, [s for s in sources if s.index in used] or sources, mode="offline")
