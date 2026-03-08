from pathlib import Path
def parse_transcript(doc_path, speakers):
    """
    Parse transcript into line-level records.

    A turn is defined as all contiguous lines spoken by the same speaker.
    If a line starts with 'SPEAKER:', that starts a new turn.
    Otherwise, the line is treated as a continuation of the current turn.

    Returns
    -------
    lines : list[dict]
        Each dict has:
        - line_id
        - turn_id
        - turn_speaker
        - line_text
    """
    with Path(doc_path).open("r", encoding = "utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    lines = []
    current_speaker = None
    current_turn_id = 0
    line_id = 0

    for raw_line in raw_lines:
        matched_speaker = None
        text = raw_line

        for speaker in speakers:
            prefix = f"{speaker}:"
            if raw_line.startswith(prefix):
                matched_speaker = speaker
                text = raw_line[len(prefix):].strip()
                break

        if matched_speaker is not None:
            current_speaker = matched_speaker
            current_turn_id += 1

        if current_speaker is not None:
            line_id += 1
            lines.append({
                "line_id": line_id,
                "turn_id": current_turn_id,
                "turn_speaker": current_speaker,
                "line_text": text,
            })

    return lines

