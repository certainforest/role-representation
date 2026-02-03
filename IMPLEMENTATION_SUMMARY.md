# Transcript to JSONL Conversion - Implementation Summary

## Overview

Successfully implemented a Python script to convert interview transcripts from plain text format to scalable JSONL entries, with one JSON object per speaker turn.

## Files Created

### 1. Conversion Script
**File:** `labs/transcript_to_jsonl.py` (395 lines)

**Components:**
- `TranscriptParser` class - Main parsing logic with regex-based speaker/timestamp detection
- `JSONLWriter` class - JSONL output generation
- CLI interface with argparse for flexible usage

**Features:**
- Automatic speaker/timestamp pattern detection
- Multi-line dialogue accumulation
- Sequential context linking (previous/next speaker)
- Metadata extraction (participants, topic, duration)
- Comprehensive validation (field presence, timestamp format, text length)
- Batch processing support
- Verbose logging option
- Validate-only mode for testing

### 2. Output Files
**Directory:** `labs/transcripts/jsonl/`

- **1.jsonl** - 37 entries (Jasmine C. & Derek Knapick, 22:42 duration)
- **2.jsonl** - 37 entries (Jasmine C. & Derek Knapick, 22:42 duration)
- **README.md** - Documentation for JSONL format and usage

## Results

### Transcripts 1 & 2: Successfully Converted ✓

**Validation Results:**
- ✓ All required fields present in every entry
- ✓ Sequential turn numbering (1-37)
- ✓ Correct previous/next speaker links
- ✓ Valid timestamp format (MM:SS)
- ✓ Non-empty dialogue text (>10 characters)

**JSON Schema Per Entry:**
```json
{
  "speaker": "Speaker Name",
  "timestamp": "MM:SS",
  "text": "Dialogue content...",
  "turn_number": 1,
  "previous_speaker": "Previous Speaker or null",
  "next_speaker": "Next Speaker or null",
  "transcript_id": "1",
  "topic": "Artificial turf vs natural grass sports injuries",
  "participants": ["Jasmine C.", "Derek Knapick"],
  "total_duration": "22:42"
}
```

**Speaker Distribution:**
- Transcript 1: Jasmine C. (19 turns), Derek Knapick (18 turns)
- Transcript 2: Jasmine C. (19 turns), Derek Knapick (18 turns)

### Transcript 3: Not Processed ⚠️

**Issue:** Source file `labs/transcripts/3.txt` is a single-line file (29,108 characters) without proper speaker/timestamp formatting.

**Action Required:** Manual formatting needed:
1. Break into separate lines
2. Add speaker names (Jasmine C., Jared, Tuan)
3. Add timestamps

**Expected Content:** Interview about open-source AI vs closed-source models

## Usage Examples

### Convert Single File
```bash
python labs/transcript_to_jsonl.py labs/transcripts/1.txt -o labs/transcripts/jsonl/1.jsonl
```

### Batch Process Directory
```bash
python labs/transcript_to_jsonl.py labs/transcripts/ -o labs/transcripts/jsonl/
```

### Validate Without Writing
```bash
python labs/transcript_to_jsonl.py labs/transcripts/1.txt --validate-only -v
```

### Override Metadata
```bash
python labs/transcript_to_jsonl.py labs/transcripts/1.txt \
  --transcript-id "interview-001" \
  --topic "Custom topic" \
  -o output.jsonl
```

## Reading JSONL Output

### Python
```python
import json

with open('labs/transcripts/jsonl/1.jsonl') as f:
    turns = [json.loads(line) for line in f]

# Access specific turn
first_turn = turns[0]
print(f"{first_turn['speaker']} ({first_turn['timestamp']}): {first_turn['text']}")
```

### Bash
```bash
# View first entry (formatted)
head -n 1 labs/transcripts/jsonl/1.jsonl | python -m json.tool

# Count turns
wc -l labs/transcripts/jsonl/*.jsonl

# Extract all speakers
jq -r '.speaker' labs/transcripts/jsonl/1.jsonl | sort -u
```

## Implementation Notes

### Parsing Strategy
1. Read file with UTF-8 encoding (fallback to latin-1)
2. Strip line numbers added by Read tool (format: `  123→`)
3. Match speaker/timestamp pattern: `^([A-Za-z\s.]+?)\s+(\d{1,2}:\d{2})\s*$`
4. Accumulate dialogue lines until next speaker detected
5. Stop at footer: "Transcribed by https://otter.ai"
6. Post-process: link speakers, add metadata

### Error Handling
- File not found errors
- Encoding fallback (UTF-8 → latin-1)
- Malformed speaker lines (logged, skipped)
- Empty dialogue (minimum 10 chars required)
- Invalid timestamps (warned but not fatal)

### Metadata Auto-Detection
- **Participants:** Unique speakers in order of first appearance
- **Total duration:** Timestamp from last turn
- **Topic:** Known topics for transcripts 1-3, keyword inference otherwise

## Verification Commands

```bash
# Run validation tests
cd labs/transcripts/jsonl

# Check all required fields
python3 << 'EOF'
import json
with open('1.jsonl') as f:
    turn = json.loads(f.readline())
    required = ['speaker', 'timestamp', 'text', 'turn_number', 'previous_speaker',
                'next_speaker', 'transcript_id', 'topic', 'participants', 'total_duration']
    missing = [f for f in required if f not in turn]
    print("✓ All fields present" if not missing else f"✗ Missing: {missing}")
EOF

# Verify turn sequence
python3 << 'EOF'
import json
with open('1.jsonl') as f:
    turns = [json.loads(line) for line in f]
    for i, t in enumerate(turns, 1):
        assert t['turn_number'] == i, f"Turn {i} mismatch"
    print(f"✓ All {len(turns)} turns sequential")
EOF

# Verify speaker links
python3 << 'EOF'
import json
with open('1.jsonl') as f:
    turns = [json.loads(line) for line in f]
    for i in range(1, len(turns) - 1):
        assert turns[i]['next_speaker'] == turns[i+1]['speaker']
        assert turns[i]['previous_speaker'] == turns[i-1]['speaker']
    print("✓ All speaker links correct")
EOF
```

## Next Steps

1. **Transcript 3:** Format source file with proper speaker/timestamp structure
2. **Scalability:** Script is ready for additional transcripts in same format
3. **Analysis:** JSONL format enables easy filtering, aggregation, and NLP processing
4. **Extensions:** Add support for alternative timestamp formats, speaker aliases, etc.

## Summary

- **Created:** 1 Python script (395 lines)
- **Generated:** 2 JSONL files (74 entries total)
- **Validated:** 100% of converted entries pass all checks
- **Pending:** 1 transcript requires source formatting
- **Ready:** For immediate use in downstream analysis pipelines
