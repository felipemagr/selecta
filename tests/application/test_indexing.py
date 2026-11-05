from pydantic import BaseModel

from selecta.application.indexing import index_collection
from selecta.application.ports import ModelOutputError
from selecta.application.retrieval import semantic_search
from selecta.domain.collection import Artist, Collection, CollectionItem, Release
from selecta.infrastructure.integrations.postgres import PostgresStore


class FakeChat:
    model = "fake-chat"

    def __init__(self) -> None:
        self.calls = 0

    def complete_json[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        self.calls += 1
        mood = "calm" if "Piano" in user else "intense"
        return schema.model_validate({"mood": mood, "energy": "low"})


class FakeEmbedder:
    model = "fake-embed"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts += texts
        vectors = []
        for text in texts:
            v = [0.0] * 1024
            for word in text.lower().split():
                v[hash(word) % 1024] += 1.0
            vectors.append(v)
        return vectors


class Source:
    def __init__(self, collection: Collection) -> None:
        self.collection = collection

    def load(self) -> Collection:
        return self.collection


def _collection(*titles: str, wanted: str | None = None) -> Collection:
    releases = [
        Release(id=i, title=t, artists=["Someone"], artist_ids=[1], genres=["Jazz"])
        for i, t in enumerate([*titles, *([wanted] if wanted else [])], start=1)
    ]
    items = [CollectionItem(release_id=r.id, owned=r.title != wanted) for r in releases]
    artists = [Artist(id=1, name="Someone", profile="A pianist from Lisbon.")]
    return Collection(releases=releases, artists=artists, items=items)


def test_reindex_only_calls_models_for_changed_records(store: PostgresStore) -> None:
    chat, embedder = FakeChat(), FakeEmbedder()
    index_collection(Source(_collection("Piano", "Drums")), store, chat, embedder)
    embedder.texts.clear()

    index_collection(Source(_collection("Piano", "Drums Live")), store, chat, embedder)

    assert chat.calls == 3
    assert len(embedder.texts) == 1 and "Drums Live" in embedder.texts[0]


def test_wantlist_records_are_not_enriched(store: PostgresStore) -> None:
    chat = FakeChat()

    index_collection(Source(_collection("Piano", wanted="Drums")), store, chat, FakeEmbedder())

    assert chat.calls == 1


def test_removed_records_leave_semantic_search(store: PostgresStore) -> None:
    chat, embedder = FakeChat(), FakeEmbedder()
    index_collection(Source(_collection("Piano", "Drums")), store, chat, embedder)

    index_collection(Source(_collection("Piano")), store, chat, embedder)

    hits = semantic_search(store, embedder, "Drums", limit=5)
    assert ("release", 2) not in {(p.kind, p.ref_id) for p in hits}
    assert ("release", 1) in {(p.kind, p.ref_id) for p in hits}


class BrokenChat(FakeChat):
    def complete_json[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        if "Drums" in user:
            raise ModelOutputError("garbled")
        return super().complete_json(system, user, schema)


def test_invalid_answer_skips_record_and_retries_next_time(store: PostgresStore) -> None:
    index_collection(Source(_collection("Piano", "Drums")), store, BrokenChat(), FakeEmbedder())
    chat = FakeChat()

    index_collection(Source(_collection("Piano", "Drums")), store, chat, FakeEmbedder())

    assert chat.calls == 1
