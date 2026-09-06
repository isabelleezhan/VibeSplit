from sqlalchemy import JSON, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class User(Base):
    """Spotify OAuth tokens + basic profile, per PROJECT.md's data model."""

    # The actual Postgres table name this class maps to.
    __tablename__ = "users"

    spotify_id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    access_token: Mapped[str] = mapped_column(String, nullable=False)
    # `str | None` is the modern way to write "optional"
    refresh_token: Mapped[str | None] = mapped_column(String, nullable=True)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False)


class TrackTagCache(Base):
    """Cached genre/mood tags per Spotify track id, so re-splitting a
    playlist (or splitting a different one that shares tracks with an
    earlier one) doesn't re-spend Gemini's free-tier quota re-tagging
    tracks it already has an answer for. See app/tracks.py for how this
    gets read/written — local files (no stable Spotify id) are never
    cached, since there's nothing stable to key them by across runs."""

    __tablename__ = "track_tag_cache"

    track_id: Mapped[str] = mapped_column(String, primary_key=True)
    genres: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    moods: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    # Unix timestamp: entries older than
    # CACHE_TTL_SECONDS (app/tracks.py) are treated as a miss.
    tagged_at: Mapped[float] = mapped_column(Float, nullable=False)
