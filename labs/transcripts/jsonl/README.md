# Transcript JSONL Files

This directory contains interview transcripts converted to JSONL (JSON Lines) format, with one JSON object per speaker turn.

## Files

- `1.jsonl` - Interview about artificial turf vs natural grass sports injuries (37 turns)
- `2.jsonl` - Interview about artificial turf vs natural grass sports injuries (37 turns)
- `3.jsonl` - **NOT GENERATED** - Source file requires formatting

## JSON Schema

Each line in the JSONL files represents one speaker turn with the following fields:

```json
{
  "speaker": "Speaker Name",
  "timestamp": "MM:SS",
  "text": "Speaker's dialogue content",
  "turn_number": 1,
  "previous_speaker": "Previous Speaker Name or null",
  "next_speaker": "Next Speaker Name or null",
  "transcript_id": "1",
  "topic": "Interview topic",
  "participants": ["Speaker 1", "Speaker 2"],
  "total_duration": "MM:SS"
}
```

## Processing Details

### Transcripts 1 & 2
- Successfully parsed from formatted text with speaker/timestamp pairs
- 37 turns each (alternating between Jasmine C. and Derek Knapick)
- Topic: Artificial turf vs natural grass sports injuries
- Duration: 22:42

### Transcript 3
**Status: Requires manual formatting**

The source file `labs/transcripts/3.txt` is a single-line file (29,108 characters) without proper speaker/timestamp formatting. It appears to be a continuous dialogue that needs to be:

1. Broken into separate lines
2. Annotated with speaker names (Jasmine C., Jared, Tuan)
3. Timestamped appropriately

The transcript appears to be about open-source AI vs closed-source models, with participants:
- Jasmine C. (NBC News AI reporter)
- Jared (AI reporter)
- Tuan (CEO of BaseTen, AI infrastructure)

## Usage

### Reading JSONL files

Python:
```python
import json

with open('1.jsonl') as f:
    turns = [json.loads(line) for line in f]
```

Bash:
```bash
# View first entry
head -n 1 1.jsonl | python -m json.tool

# Count turns
wc -l 1.jsonl
```

### Converting new transcripts

```bash
# Single file
python labs/transcript_to_jsonl.py labs/transcripts/1.txt -o labs/transcripts/jsonl/1.jsonl

# Batch process
python labs/transcript_to_jsonl.py labs/transcripts/ -o labs/transcripts/jsonl/

# Validate only
python labs/transcript_to_jsonl.py labs/transcripts/1.txt --validate-only -v
```

## Validation

All generated JSONL files have been validated for:
- ✓ All required fields present
- ✓ Sequential turn numbering
- ✓ Correct previous/next speaker links
- ✓ Valid timestamp format (MM:SS)
- ✓ Non-empty dialogue text (minimum 10 characters)
