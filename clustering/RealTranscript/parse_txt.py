import re

def parse_transcript(text: str) -> list[tuple[str, str]]:
    """
    Parse Otter.ai style transcript into (speaker, utterance) tuples.
    """
    # Pattern: "Speaker Name  HH:MM:SS" or "Speaker Name  HH:MM"
    pattern = r'^(.+?)\s{2,}(\d{1,2}:\d{2}(?::\d{2})?)\s*$'
    
    lines = text.strip().split('\n')
    transcript = []
    current_speaker = None
    current_text = []
    
    for line in lines:
        match = re.match(pattern, line)
        if match:
            # Save previous speaker's text
            if current_speaker and current_text:
                utterance = ' '.join(current_text).strip()
                if utterance:
                    transcript.append((current_speaker, utterance))
            
            # Start new speaker
            current_speaker = match.group(1).strip()
            current_text = []
        else:
            # Continuation of current speaker's text
            if line.strip() and not line.startswith('Transcribed by'):
                current_text.append(line.strip())
    
    # Don't forget last speaker
    if current_speaker and current_text:
        utterance = ' '.join(current_text).strip()
        if utterance:
            transcript.append((current_speaker, utterance))
    
    return transcript


import re

def parse_transcript_sentences(text: str) -> list[tuple[str, str]]:
    """
    Parse Otter.ai style transcript into (speaker, sentence) tuples.
    Splits each utterance into individual sentences.
    """
    # Pattern: "Speaker Name  HH:MM:SS" or "Speaker Name  HH:MM"
    pattern = r'^(.+?)\s{2,}(\d{1,2}:\d{2}(?::\d{2})?)\s*$'
    
    lines = text.strip().split('\n')
    transcript = []
    current_speaker = None
    current_text = []
    
    def split_sentences(text: str) -> list[str]:
        # Split on . ! ? followed by space or end of string
        # Keeps abbreviations like "U.S." or "Dr." mostly intact
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    for line in lines:
        match = re.match(pattern, line)
        if match:
            # Save previous speaker's sentences
            if current_speaker and current_text:
                utterance = ' '.join(current_text).strip()
                for sentence in split_sentences(utterance):
                    transcript.append((current_speaker, sentence))
            
            # Start new speaker
            current_speaker = match.group(1).strip()
            current_text = []
        else:
            if line.strip() and not line.startswith('Transcribed by'):
                current_text.append(line.strip())
    
    # Don't forget last speaker
    if current_speaker and current_text:
        utterance = ' '.join(current_text).strip()
        for sentence in split_sentences(utterance):
            transcript.append((current_speaker, sentence))
    
    return transcript

import re

def parse_debate_transcript_sentences(text: str) -> list[tuple[str, str]]:
    """
    Parse debate transcript using known speakers from header.
    """
    # Extract speakers from header line
    speaker_match = re.search(r'SPEAKER:\s*\[([^\]]+)\]', text)
    speakers = [s.strip() for s in speaker_match.group(1).split(';')]
    
    # Build pattern: HOLT:|CLINTON:|TRUMP:
    pattern = '(' + '|'.join(speakers) + '):'
    
    # Split by speaker labels
    parts = re.split(pattern, text)
    
    # parts = ['preamble', 'HOLT', ' text...', 'CLINTON', ' text...', ...]
    transcript = []
    
    def split_sentences(text: str) -> list[str]:
        text = re.sub(r'\[.*?\]', '', text)  # remove [applause] etc
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    for i in range(1, len(parts) - 1, 2):
        speaker = parts[i]
        text = parts[i + 1].strip()
        for sentence in split_sentences(text):
            transcript.append((speaker, sentence))
    
    return transcript

def parse_debate_transcript_blocks(text: str) -> list[tuple[str, str]]:
    """
    Parse debate transcript into (speaker, full_utterance) tuples.
    No sentence splitting.
    """
    speaker_match = re.search(r'SPEAKER:\s*\[([^\]]+)\]', text)
    speakers = [s.strip() for s in speaker_match.group(1).split(';')]
    
    pattern = '(' + '|'.join(speakers) + '):'
    parts = re.split(pattern, text)
    
    transcript = []
    for i in range(1, len(parts) - 1, 2):
        speaker = parts[i]
        text = re.sub(r'\[.*?\]', '', parts[i + 1]).strip()  # remove [applause] etc
        if text:
            transcript.append((speaker, text))
    
    return transcript


# # Usage
# with open("2016debate.txt", "r") as f:
#     raw = f.read()

# transcript = parse_debate_transcript(raw)

# for speaker, text in transcript[50:80]:
#     print(f"{speaker}: {text[:100]}...")