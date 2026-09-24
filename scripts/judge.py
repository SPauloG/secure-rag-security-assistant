import openai
import voyageai
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from src import config


class _VoyageEmbeddings(BaseRagasEmbedding):
    """RAGAS has no Voyage provider; this plugs Voyage into its embedding interface."""

    def __init__(self):
        super().__init__()
        self._client = voyageai.Client(api_key=config.VOYAGE_API_KEY)

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self._client.embed(
            [text],
            model=config.EMBEDDING_MODEL,
            input_type="query",  # answer_relevancy compares questions with questions
            output_dimension=config.EMBEDDING_DIMENSION,
        ).embeddings[0]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text)


GEMINI_OPENAI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _judge_llm():
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing in .env (needed by the evaluation judge).")
    # Gemini's OpenAI-compatible endpoint, so RAGAS uses its default OpenAI path
    # free tier allows few requests per minute; retries back off on 429 until it frees up
    client = openai.AsyncOpenAI(
        api_key=config.GEMINI_API_KEY, base_url=GEMINI_OPENAI_ENDPOINT, max_retries=8
    )
    # room for Gemini's thinking tokens, which count toward max_tokens
    return llm_factory(config.JUDGE_MODEL, provider="openai", client=client, max_tokens=8192)


class Judge:
    def __init__(self):
        llm = _judge_llm()
        self._faithfulness = Faithfulness(llm=llm)
        self._relevancy = AnswerRelevancy(llm=llm, embeddings=_VoyageEmbeddings())

    def score(self, question: str, answer: str, chunks: list[dict]) -> dict[str, float]:
        answer = answer.split("\n\nSources:\n")[0]  # judge the prose, not the citation list
        return {
            "faithfulness": self._faithfulness.score(
                user_input=question, response=answer, retrieved_contexts=[c["text"] for c in chunks]
            ).value,
            "answer_relevancy": self._relevancy.score(user_input=question, response=answer).value,
        }
