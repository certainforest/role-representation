import re

def convert_debate_transcript(input_path: str, output_path: str):
    """
    Convert debate transcript to simple 'SPEAKER: text' format.
    One line per utterance.
    """
    with open(input_path) as f:
        text = f.read()
    
    # Extract speakers from header
    speaker_match = re.search(r'SPEAKER:\s*\[([^\]]+)\]', text)
    if not speaker_match:
        raise ValueError("No SPEAKER: [...] header found")
    
    speakers = [s.strip() for s in speaker_match.group(1).split(';')]
    print(f"Found speakers: {speakers}")
    
    # Split on speaker names
    pattern = r'\b(' + '|'.join(re.escape(s) for s in speakers) + r'):'
    parts = re.split(pattern, text)
    
    with open(output_path, 'w') as f:
        for i in range(1, len(parts) - 1, 2):
            speaker = parts[i]
            utterance = parts[i + 1]
            
            # Remove stage directions
            utterance = re.sub(r'\([^)]*\)', '', utterance)  # (APPLAUSE)
            utterance = re.sub(r'\[[^\]]*\]', '', utterance)  # [crosstalk]
            
            # Collapse whitespace to single spaces
            utterance = ' '.join(utterance.split())
            
            if utterance:
                f.write(f"{speaker}: {utterance}\n")
    
    print(f"✅ Saved to {output_path}")

def transcript_shuffle(input_path: str, output_path: str):
    """
    Shuffle lines in a transcript file.
    """
    import random

    # -------------------
    # Load transcript
    # -------------------
    lines = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)

    print(f"Loaded {len(lines)} lines.")

    # -------------------
    # Shuffle lines
    # -------------------
    shuffled_lines = lines.copy()
    random.shuffle(shuffled_lines)

    # -------------------
    # Write shuffled transcript
    # -------------------
    with open(output_path, "w") as f:
        for line in shuffled_lines:
            f.write(line + "\n")

    print(f"Shuffled transcript saved to {output_path}")

# Usage
# convert_debate_transcript("/mnt/ssd/aryawu/role-representation/clustering/RealTranscript/2008debate.txt", "2008debate_clean.txt")
transcript_shuffle("transcript1.txt", "transcript1_shuffled.txt")