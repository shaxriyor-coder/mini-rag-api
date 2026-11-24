from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app import llm  # noqa: F401  (you will need it)
from app.chunking import chunk_text  # noqa: F401  (you will need it)


class DocumentInput(BaseModel):
    id: str
    text: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id must not be empty")
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


class IngestInput(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1)


class AskInput(BaseModel):
    question: str
    top_k: Annotated[int, Field(ge=1, le=10)] = 3

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value


def create_app() -> FastAPI:
    """Return a NEW app with an EMPTY in-memory store.

    The tests call create_app() once per test, so state must not leak
    between app instances (no module-level store).
    """
    app = FastAPI()
    documents: dict[str, list[dict]] = {}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ingest")
    def ingest(payload: IngestInput):
        total_chunks = 0

        for document in payload.documents:
            chunks = chunk_text(document.text)
            stored_chunks: list[dict] = []

            for index, text in enumerate(chunks):
                chunk_id = f"{document.id}:{index}"
                stored_chunks.append(
                    {
                        "doc_id": document.id,
                        "chunk_id": chunk_id,
                        "text": text,
                        "vector": llm.embed(text),
                    }
                )

            documents[document.id] = stored_chunks
            total_chunks += len(chunks)

        return {"ingested": len(payload.documents), "chunks": total_chunks}

    @app.post("/ask")
    def ask(payload: AskInput):
        if not documents:
            raise HTTPException(status_code=404, detail="No documents ingested")

        question_vector = llm.embed(payload.question)
        scored_chunks: list[tuple[float, dict]] = []

        for chunks in documents.values():
            for chunk in chunks:
                score = sum(
                    question_value * chunk_value
                    for question_value, chunk_value in zip(
                        question_vector, chunk["vector"]
                    )
                )
                scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        best_chunks = scored_chunks[: payload.top_k]

        contexts = [chunk["text"] for _, chunk in best_chunks]
        sources = [
            {
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "score": score,
                "text": chunk["text"],
            }
            for score, chunk in best_chunks
        ]

        return {
            "answer": llm.generate(payload.question, contexts),
            "sources": sources,
        }

    @app.get("/documents")
    def list_documents():
        return {
            "documents": [
                {"id": doc_id, "chunks": len(documents[doc_id])}
                for doc_id in sorted(documents)
            ]
        }

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str):
        if doc_id not in documents:
            raise HTTPException(status_code=404, detail="Document not found")
        del documents[doc_id]
        return None

    return app


app = create_app()
