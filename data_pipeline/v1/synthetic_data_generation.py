"""
Speaker Attribution Dataset Generator
Generates minimal pairs for levels 1, 2, and 2.5 using OpenAI API
"""

import json
import random
from openai import OpenAI

client = OpenAI()  # Set OPENAI_API_KEY env variable

# ============================================================
# PROMPTS FOR EACH LEVEL
# ============================================================

LEVEL_1_2_PROMPT = """Generate {n} conversation snippets for a speaker attribution task.

Each example should have EXACTLY this structure:
- Line 1: Speaker introduces themselves with name
- Line 2: Second speaker introduces themselves with name  
- Line 3: First speaker states a preference/belief about a topic
- Line 4: Second speaker states a DIFFERENT preference/belief about the SAME topic

Requirements:
- Use diverse name pairs (vary gender, cultural background)
- Use diverse topics: pets, sports, food, colors, music genres, seasons, hobbies, school subjects, drinks, transportation, movies, books, weather, time of day, holidays
- Answers must be single words or 2-3 word phrases maximum
- The two preferences must be clearly distinct and unambiguous
- Make preferences natural and realistic

Output as a JSON array with this exact structure:
[
  {{
    "names": ["Name1", "Name2"],
    "topic": "topic category",
    "transcript_unlabeled": [
      "I am Name1.",
      "I am Name2.",
      "I [preference statement].",
      "I [different preference statement]."
    ],
    "beliefs": {{
      "Name1": "answer1",
      "Name2": "answer2"
    }}
  }}
]

Generate exactly {n} examples with maximum variety. Output ONLY valid JSON, no other text."""


LEVEL_2_5_PROMPT = """Generate {n} three-person conversation snippets for a speaker attribution task.

Each conversation must:
1. Have exactly 3 speakers who all state preferences about a shared topic
2. Use NON-round-robin turn taking (speakers don't just alternate A-B-C-A-B-C)
3. Be uniquely SOLVABLE via pragmatic turn-taking and discourse constraints, including these disambiguation cues:
   - Self-introductions anchor identity
   - Round-robin assumption for early turns
   - Third-party mentions (e.g., "I heard [Name] likes X too") 
   - Vocative address selects next speaker (e.g., "Hi [Name]!")
   - Questions select next speaker (asking something one already knows the answer to → directed at someone else)
   - Agreement/confirmation of statements made about oneself
4. CRITICAL CONSTRAINT: Two consecutive lines are NEVER spoken by the same speaker. Every line is a new speaker.

EXAMPLE with full reasoning:
Transcript:
Line 1: "I am Alice."
Line 2: "I am Bob."
Line 3: "I like basketball."
Line 4: "I heard that Claire likes basketball too!"
Line 5: "Hi Alice! Yes, I like basketball. But I like tennis more!"
Line 6: "Fun! What do you like?"
Line 7: "I like soccer."

Turn sequence: ["Alice", "Bob", "Alice", "Bob", "Claire", "Alice", "Bob"]

Disambiguation reasoning:
- [Line 1] Alice: Self-introduction anchor
- [Line 2] Bob: Self-introduction anchor, must be different speaker from Line 1
- [Line 3] Alice: Round-robin assumption (back to first speaker)
- [Line 4] Bob: Round-robin assumption (back to second speaker), introduces Claire as third party.
- [Line 5] Claire: Claire confirms "Yes, I like basketball". Claire greets Alice, selecting Alice for the next turn.
- [Line 6] Alice: Alice responds to Claire with "Fun!". Alice then question "What do you like?", which must be directed at Bob since Claire already stated her answer to this question.
- [Line 7] Bob: This line must be Bob answering Alice's question, since only Bob is left to answer the question about his preference on sports.

QA pairs:
- "What sport does Alice like the most?" → "basketball"
- "What sport does Bob like the most?" → "soccer"  
- "What sport does Claire like the most?" → "tennis"

Generate {n} NEW examples following this pattern. Each must:
- Have 5-8 turns total
- Include at least one non-round-robin transition (like Claire entering, or turn selection via vocative/question)
- Be fully solvable using the disambiguation cues above
- Have each speaker state a clear, distinct preference

Output as a JSON array: Here is the structure for another example:
[
  {{
    "names": ["Emma", "Liam", "Noah"],
    "topic": "season", 
    "transcript_unlabeled": [
      "I am Emma.",
      "I am Liam.",
      "I think summer is the best season.",
      "I disagree, winter is better. Noah, what do you think?",
      "I actually prefer autumn!",
      "Autumn is nice! Emma, don't you like autumn?",
      "I do but i love the beach more!"
    ],
    "turn_sequence": ["Emma", "Liam", "Emma", "Liam", "Noah", "Liam", "Emma"],
    "disambiguation_reasoning": [
      "[Line 1] Emma: Self-introduction anchor",
      "[Line 2] Liam: Self-introduction anchor, must be different speaker from Line 1",
      "[Line 3] Emma: Round-robin assumption (back to first speaker)",
      "[Line 4] Liam: Round-robin assumption (back to second speaker), introduces Noah as third party",
      "[Line 5] Noah: Speaker answers the question posed by Liam about Noah's preference", so this must be Noah entering the conversation",
      "[Line 6] Liam: Liam responds to Noah's answer to his question. He addresses Emma with 'Emma, don't you like autumn?', selecting Emma for the next turn",
      "[Line 7] Emma: Emma responds to Liam's question about her preference",
    ],
    "beliefs": {{
      "Emma": "summer",
      "Liam": "winter",
      "Noah": "autumn"
    }}
  }}
]

Vary the topics (food, music, movies, hobbies, travel, books, etc.) and names.
Generate exactly {n} examples. Output ONLY valid JSON, no other text."""


# ============================================================
# GENERATION FUNCTIONS
# ============================================================

def query_gpt(prompt, model="gpt-5"):
    """Query OpenAI API and return parsed JSON."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates discourse datasets. Always output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
    )
    content = response.choices[0].message.content
    # Clean up potential markdown code blocks
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def generate_level_1_2(n=50, batch_size=10):
    """Generate Level 1 & 2 examples (same content, different labeling)."""
    all_examples = []
    
    for batch in range(n // batch_size):
        print(f"Generating Level 1/2 batch {batch + 1}/{n // batch_size}...")
        prompt = LEVEL_1_2_PROMPT.format(n=batch_size)
        examples = query_gpt(prompt)
        all_examples.extend(examples)
    
    # Format into Level 1 (labeled) and Level 2 (unlabeled) versions
    dataset = []
    for ex in all_examples:
        name1, name2 = ex["names"]
        
        # Build labeled version
        transcript_labeled = [
            f"{name1}: {ex['transcript_unlabeled'][0]}",
            f"{name2}: {ex['transcript_unlabeled'][1]}",
            f"{name1}: {ex['transcript_unlabeled'][2]}",
            f"{name2}: {ex['transcript_unlabeled'][3]}"
        ]
        
        # Build QA pairs
        qa_pairs = [
            {
                "prompt": f"What does {name1} prefer? Answer: {name1} prefers",
                "prompt_instruct": f"What does {name1} prefer?",
                "groundtruth": ex["beliefs"][name1]
            },
            {
                "prompt": f"What does {name2} prefer? Answer: {name2} prefers",
                "prompt_instruct": f"What does {name2} prefer?",
                "groundtruth": ex["beliefs"][name2]
            }
        ]
        
        dataset.append({
            "level": "1_and_2",
            "topic": ex["topic"],
            "names": ex["names"],
            "transcript_labeled": transcript_labeled,
            "transcript_unlabeled": ex["transcript_unlabeled"],
            "qa_pairs": qa_pairs,
            "beliefs": ex["beliefs"]
        })
    
    return dataset


def generate_level_2_5(n=50, batch_size=5):
    """Generate Level 2.5 examples (3 speakers, non-round-robin)."""
    all_examples = []
    
    for batch in range(n // batch_size):
        print(f"Generating Level 2.5 batch {batch + 1}/{n // batch_size}...")
        prompt = LEVEL_2_5_PROMPT.format(n=batch_size)
        try:
            examples = query_gpt(prompt)
            all_examples.extend(examples)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error in batch {batch + 1}, retrying...")
            try:
                examples = query_gpt(prompt)
                all_examples.extend(examples)
            except:
                print(f"  Skipping batch {batch + 1}")
                continue
    
    # Format into final structure
    dataset = []
    for ex in all_examples:
        names = ex["names"]
        
        # Build labeled version using turn_sequence
        transcript_labeled = []
        for i, line in enumerate(ex["transcript_unlabeled"]):
            if i < len(ex.get("turn_sequence", [])):
                speaker = ex["turn_sequence"][i]
                transcript_labeled.append(f"{speaker}: {line}")
            else:
                transcript_labeled.append(line)
        
        # Build QA pairs for all 3 speakers
        qa_pairs = []
        for name in names:
            if name in ex["beliefs"]:
                qa_pairs.append({
                    "prompt": f"What does {name} prefer? Answer: {name} prefers",
                    "prompt_instruct": f"What does {name} prefer?",
                    "groundtruth": ex["beliefs"][name]
                })
        
        dataset.append({
            "level": "2.5",
            "topic": ex.get("topic", "unknown"),
            "names": names,
            "transcript_labeled": transcript_labeled,
            "transcript_unlabeled": ex["transcript_unlabeled"],
            "turn_sequence": ex.get("turn_sequence", []),
            "disambiguation_reasoning": ex.get("disambiguation_reasoning", []),
            "qa_pairs": qa_pairs,
            "beliefs": ex["beliefs"]
        })
    
    return dataset


def validate_example(example):
    """Basic validation that example is well-formed."""
    issues = []
    
    # Check all names have beliefs
    for name in example["names"]:
        if name not in example["beliefs"]:
            issues.append(f"Missing belief for {name}")
    
    # Check QA pairs match beliefs
    for qa in example["qa_pairs"]:
        if not qa["groundtruth"]:
            issues.append("Empty groundtruth")
    
    # For 2.5, check turn sequence length
    if example["level"] == "2.5":
        if len(example.get("turn_sequence", [])) != len(example["transcript_unlabeled"]):
            issues.append("Turn sequence length mismatch")
    
    return issues


def generate_full_dataset(n_per_level=50):
    """Generate complete dataset for all levels."""
    print("=" * 50)
    print("Generating Speaker Attribution Dataset")
    print("=" * 50)
    
    # Generate Level 1/2
    print("\n[Level 1 & 2: 2 speakers, labeled vs unlabeled]")
    level_1_2 = generate_level_1_2(n=n_per_level)
    print(f"Generated {len(level_1_2)} examples")
    
    # Generate Level 2.5
    print("\n[Level 2.5: 3 speakers, non-round-robin]")
    level_2_5 = generate_level_2_5(n=n_per_level)
    print(f"Generated {len(level_2_5)} examples")
    
    # Validate
    print("\n[Validating...]")
    all_examples = level_1_2 + level_2_5
    valid_examples = []
    for ex in all_examples:
        issues = validate_example(ex)
        if not issues:
            valid_examples.append(ex)
        else:
            print(f"  Dropping example: {issues}")
    
    print(f"Valid examples: {len(valid_examples)}/{len(all_examples)}")
    
    # Save
    dataset = {
        "metadata": {
            "description": "Speaker attribution dataset for pragmatic inference evaluation",
            "levels": {
                "1_and_2": "2 speakers, minimal structure, labeled vs unlabeled comparison",
                "2.5": "3 speakers, non-round-robin, requires discourse cue tracking"
            },
            "total_examples": len(valid_examples)
        },
        "examples": valid_examples
    }
    
    with open("speaker_attribution_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nSaved to speaker_attribution_dataset.json")
    return dataset


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    dataset = generate_full_dataset(n_per_level=10)
    
    # Print sample
    print("\n" + "=" * 50)
    print("SAMPLE OUTPUTS")
    print("=" * 50)
    
    print("\n--- Level 1/2 Example ---")
    l12_ex = [e for e in dataset["examples"] if e["level"] == "1_and_2"][0]
    print(f"Topic: {l12_ex['topic']}")
    print(f"Labeled:\n  " + "\n  ".join(l12_ex["transcript_labeled"]))
    print(f"Unlabeled:\n  " + "\n  ".join(l12_ex["transcript_unlabeled"]))
    print(f"QA: {l12_ex['qa_pairs']}")
    
    print("\n--- Level 2.5 Example ---")
    l25_ex = [e for e in dataset["examples"] if e["level"] == "2.5"][0]
    print(f"Topic: {l25_ex['topic']}")
    print(f"Turn sequence: {l25_ex['turn_sequence']}")
    print(f"Transcript:\n  " + "\n  ".join(l25_ex["transcript_unlabeled"]))
    print(f"Disambiguation reasoning:")
    for r in l25_ex.get("disambiguation_reasoning", []):
        print(f"    {r}")
    print(f"Beliefs: {l25_ex['beliefs']}")