"""Loading a collection into the store, with enrichment and embeddings."""

import hashlib
import logging

from selecta.application.ports import (
    ChatModel,
    Chunk,
    ChunkKind,
    CollectionSource,
    CollectionStore,
    Embedder,
    ModelOutputError,
)
from selecta.application.prompts import MOOD_ENRICH
from selecta.domain.collection import Artist, Release
from selecta.domain.enrichment import Enrichment

logger = logging.getLogger(__name__)

MAX_NOTES_CHARS = 1000
MAX_PROFILE_CHARS = 1000
MAX_TRACKS = 20


def index_collection(
    source: CollectionSource, store: CollectionStore, chat: ChatModel, embedder: Embedder
) -> None:
    """Replace the stored collection with the source's, then enrich and embed what changed.

    Results are cached by input hash, so a rerun only calls the models for new or changed text.
    """
    collection = source.load()
    store.replace(collection)
    logger.info(
        "indexed %d releases and %d artists", len(collection.releases), len(collection.artists)
    )
    owned = {i.release_id for i in collection.items if i.owned}
    releases = [r for r in collection.releases if r.id in owned]
    artist_ids = {a for r in releases for a in r.artist_ids}
    artists = [a for a in collection.artists if a.id in artist_ids]
    _enrich(releases, artists, store, chat)
    _embed(releases, artists, store, embedder)


def _enrich(
    releases: list[Release], artists: list[Artist], store: CollectionStore, chat: ChatModel
) -> None:
    profiles = {a.id: a.profile for a in artists if a.profile}
    stored = store.enrichment_hashes()
    for release in releases:
        user = _enrich_input(release, profiles)
        input_hash = _hash(chat.model, MOOD_ENRICH, user)
        if stored.get(release.id) == input_hash:
            continue
        try:
            enrichment = chat.complete_json(MOOD_ENRICH, user, Enrichment)
        except ModelOutputError as e:
            logger.warning("skipped %s, retried on the next index: %s", release.title, e)
            continue
        store.save_enrichment(release.id, enrichment, input_hash)
        logger.info(
            "enriched %s - %s: %s, %s",
            ", ".join(release.artists),
            release.title,
            enrichment.mood,
            enrichment.energy,
        )


def _embed(
    releases: list[Release], artists: list[Artist], store: CollectionStore, embedder: Embedder
) -> None:
    texts: dict[tuple[ChunkKind, int], str] = {
        ("artist", a.id): f"{a.name}. {a.profile}" for a in artists if a.profile
    }
    for r in releases:
        texts["release", r.id] = _release_text(r)
    stored = store.chunk_hashes()
    hashed = [(key, text, _hash(embedder.model, text)) for key, text in texts.items()]
    todo = [(key, text, h) for key, text, h in hashed if stored.get(key) != h]
    embeddings = embedder.embed([text for _, text, _ in todo]) if todo else []
    store.save_chunks(
        [
            Chunk(kind=kind, ref_id=ref_id, text=text, text_hash=text_hash, embedding=e)
            for ((kind, ref_id), text, text_hash), e in zip(todo, embeddings, strict=True)
        ]
    )
    store.delete_chunks_except(set(texts))
    logger.info("embedded %d new or changed chunks, %d in total", len(todo), len(texts))


def _release_text(release: Release) -> str:
    """Return the Discogs facts about a release as one passage, without LLM-written text."""
    year = release.original_year or release.year
    parts = [
        f"{', '.join(release.artists)} - {release.title}{f' ({year})' if year else ''}.",
        f"{', '.join([*release.genres, *release.styles])}.",
    ]
    if release.label:
        parts.append(f"{release.label}.")
    if release.notes:
        parts.append(release.notes[:MAX_NOTES_CHARS])
    return " ".join(parts)


def _enrich_input(release: Release, profiles: dict[int, str]) -> str:
    lines = [
        f"Artists: {', '.join(release.artists)}",
        f"Title: {release.title}",
        f"Year: {release.original_year or release.year or 'unknown'}",
        f"Label: {release.label or 'unknown'}",
        f"Genres: {', '.join(release.genres)}",
        f"Styles: {', '.join(release.styles)}",
        f"Tracks: {', '.join(t.title for t in release.tracks[:MAX_TRACKS])}",
    ]
    if release.notes:
        lines.append(f"Notes: {release.notes[:MAX_NOTES_CHARS]}")
    for artist_id in release.artist_ids:
        if artist_id in profiles:
            lines.append(f"Artist profile: {profiles[artist_id][:MAX_PROFILE_CHARS]}")
    return "\n".join(lines)


def _hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()
