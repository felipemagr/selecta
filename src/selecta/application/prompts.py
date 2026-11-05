"""Every prompt the application sends to an LLM."""

MOOD_ENRICH = """\
You tag records in a listener's vinyl collection by how they feel, so the right one can be \
picked for the moment. You get the Discogs data for one record.

Return:
- mood: the closest of calm, warm, melancholic, upbeat, intense, dark.
- energy: low, medium or high.

Judge from the data given and what you reliably know about the record. If you don't recognise \
it and the data is too thin to judge, use UNKNOWN."""
