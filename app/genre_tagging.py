"""LLM-based genre + mood/energy tagging. Tags every track.

Tagging happens in fixed-size chunks rather than one call for the whole
playlist. Chunks are processed sequentially (not concurrently) so each one 
can be told which genre/mood tags earlier chunks already used — otherwise 
a later chunk might independently invent "indie-rock" where an earlier one 
used "indie rock".

Two separate tag dimensions come out of this:
  - genre tags: the musical genre itself ("pop", "hip hop", "house")
  - mood tags: energy/setting/vibe ("high-energy", "rave", "chill",
    "workout", "late-night")

Tagging is per-track rather than per-artist. The prompt still asks the model 
to default to an artist's usual sound/mood and stay consistent across their 
tracks.

Uses Google's Gemini API (free tier) rather than a paid-only provider.
"""

from pydantic import BaseModel

from app.llm import generate_structured

CHUNK_SIZE = 40 # Tracks per LLM call

_SYSTEM_PROMPT = (
    "You are a music metadata assistant. Given a list of tracks (each with a title "
    "and artist name(s)), assign each track two things:\n"
    "1. `genre_tags`: 2-4 lowercase genre tags (e.g. 'pop', 'r&b', 'hip hop', 'k-pop', "
    "'house', 'indie rock', 'singer-songwriter', 'trap', 'boy band').\n"
    "2. `mood_tags`: 1-3 lowercase mood/energy/setting tags describing the vibe of "
    "the song itself, independent of genre (e.g. 'high-energy', 'rave', 'chill', "
    "'workout', 'romantic', 'late-night', 'melancholy', 'summery', 'aggressive'). Two "
    "tracks can share the same genre_tags but have very different mood_tags — that's "
    "expected and important (a chill downtempo house track vs. a peak-time rave "
    "house track are both 'house' but very different moods).\n\n"
    "For both, default to an artist's usual sound/mood and use the SAME tags across "
    "that artist's tracks — only give a track different tags when it's a clear, "
    "confident departure from the artist's other tracks in this list (e.g. a rap "
    "artist's R&B feature, a band's acoustic version, a producer's one uncharacteristically "
    "mellow track). Don't invent a different tag spelling/wording for the same "
    "genre or mood across tracks or artists — consistent, reused tags are what make "
    "these useful for grouping similar tracks together later. If you don't recognize "
    "a track or artist, make a reasonable guess from the title/name/context rather "
    "than leaving tags empty."
)


class _TrackTags(BaseModel):
    track_key: str
    genre_tags: list[str]
    mood_tags: list[str]


class _TaggingResponse(BaseModel):
    tracks: list[_TrackTags]


def _format_track(track: dict) -> str:
    artists = ", ".join(a["name"] for a in track.get("artists", []) if a.get("name")) or "Unknown Artist"
    name = track.get("name") or "Untitled"
    return f"- [{track['_tag_key']}] \"{name}\" by {artists}"


def _build_prompt(tracks: list[dict], known_genre_tags: set[str], known_mood_tags: set[str]) -> str:
    lines = []
    if known_genre_tags or known_mood_tags:
        lines.append(
            "Tags already used elsewhere in this playlist — reuse these exact "
            "spellings when they apply, rather than inventing a different wording "
            "for the same genre or mood:\n"
            f"  genres so far: {', '.join(sorted(known_genre_tags)) or '(none yet)'}\n"
            f"  moods so far: {', '.join(sorted(known_mood_tags)) or '(none yet)'}\n"
        )
    lines.append("Tracks:")
    lines.extend(_format_track(t) for t in tracks)
    return "\n".join(lines)


async def tag_tracks(tracks: list[dict]) -> dict[str, dict[str, list[str]]]:
    """Takes tracks with a `_tag_key` (see get_enriched_playlist_tracks —
    the Spotify track id, or a positional fallback for local files that
    often lack one), `name`, and `artists`. Returns
    {tag_key: {"genres": [...], "moods": [...]}} for every track given.

    Processes CHUNK_SIZE tracks per LLM call, sequentially — see the
    module docstring for why this isn't one call for everything, and why
    it isn't chunks run concurrently either.
    """
    if not tracks:
        return {}

    results: dict[str, dict[str, list[str]]] = {}
    known_genre_tags: set[str] = set()
    known_mood_tags: set[str] = set()

    for start in range(0, len(tracks), CHUNK_SIZE):
        chunk = tracks[start : start + CHUNK_SIZE]
        contents = _build_prompt(chunk, known_genre_tags, known_mood_tags)
        parsed: _TaggingResponse = await generate_structured(_SYSTEM_PROMPT, contents, _TaggingResponse)

        for entry in parsed.tracks:
            if not entry.track_key:
                continue
            genres = [t.lower() for t in entry.genre_tags]
            moods = [t.lower() for t in entry.mood_tags]
            results[entry.track_key] = {"genres": genres, "moods": moods}
            known_genre_tags.update(genres)
            known_mood_tags.update(moods)

    return results
