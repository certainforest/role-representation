"""
Generate a long two-speaker transcript using AutoGen agents.

Usage:
    export OPENAI_API_KEY=sk-...
    python generate_transcript_autogen.py
    python generate_transcript_autogen.py --turns 300 --model gpt-4o-mini

Output format (same as transcript1.txt):
    Alice: <utterance>
    Bob: <utterance>
"""

import argparse
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ══════════════════════════════════════════════════════════════════════
# PERSONA DEFINITIONS
# ══════════════════════════════════════════════════════════════════════

STARTING_TRANSCRIPT_PATH = "transcript1_serious.txt"

def load_transcript(path):
    """Load existing transcript into list of (speaker, content) tuples."""
    history = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for name in ["Alice", "Bob"]:
                if line.startswith(f"{name}:"):
                    content = line[len(name) + 1:].strip()
                    history.append((name, content))
                    break
    return history


#"daygoing": "how is their day going", 
TOPIC = {"career": "their plan after graduation"}
# TOPIC = {"sport": "their favorite sports", "weekend": "their weekend plans", "classes": "their classes in school", "career": "their plan after graduation"}

async def run(args):
    client = OpenAIChatCompletionClient(model=args.model)

    # Load existing transcript as seed context
    history = load_transcript(STARTING_TRANSCRIPT_PATH)
    seed = "\n".join(f"{name}: {text}" for name, text in history)
    seed += "\n\nContinue this conversation. Stay in character."

    # Figure out who speaks next (opposite of last speaker)
    last_speaker = history[-1][0] if history else "Bob"
    first_agent = "Alice" if last_speaker == "Bob" else "Bob"

    for topic_name, topic_desc in TOPIC.items():
        print(f"\n=== Generating transcript about topic: {topic_name} ({topic_desc}) ===\n")

        # ALICE_SYSTEM = (
        #     f"You are Alice in a casual conversation with Bob about {topic_desc}. "
        #     "You are fun, quirky, and playful. You speak in informal, slang language. "
        #     "Keep each reply to 1-3 sentences. "
        #     "Do NOT use emojis. "
        #     "Do NOT prefix your reply with your name — just say your line directly."
        # )
        ALICE_SYSTEM = (
            f"You are Alice in a casual conversation with Bob about {topic_desc}. "
            "You are serious, precise, and methodical. You speak in clear, formal, structured sentences. "
            "Keep each reply to 1-3 sentences. "
            "Do NOT use emojis. "
            "Do NOT prefix your reply with your name — just say your line directly."
        )

        BOB_SYSTEM = (
            f"You are Bob in a casual conversation with Alice about {topic_desc}. "
            "You are serious, precise, and methodical. You speak in clear, formal, structured sentences. "
            "Keep each reply to 1-3 sentences. "
            "Do NOT use emojis. "
            "Do NOT prefix your reply with your name — just say your line directly."
        )

        alice = AssistantAgent(
            name="Alice",
            system_message=ALICE_SYSTEM,
            model_client=client,
        )

        bob = AssistantAgent(
            name="Bob",
            system_message=BOB_SYSTEM,
            model_client=client,
        )

        # Order so the correct speaker goes next
        if first_agent == "Alice":
            participants = [alice, bob]
        else:
            participants = [bob, alice]

        termination = MaxMessageTermination(max_messages=args.turns)

        team = RoundRobinGroupChat(
            participants=participants,
            termination_condition=termination,
        )

        print(f"Loaded {len(history)} turns from {STARTING_TRANSCRIPT_PATH}")
        print(f"Next speaker: {first_agent}")
        print(f"Running {args.turns} new turns with {args.model}")
        print("-" * 60)

        turn_count = len(history)

        async for message in team.run_stream(task=seed):
            # Each streamed event is either a TaskResult or an agent message
            if hasattr(message, "source") and hasattr(message, "content"):
                speaker = message.source
                content = str(message.content).strip()
                if not content:
                    continue
                # Skip the seed message (echoed back as user task)
                if "Continue this conversation" in content:
                    continue
                # Strip duplicate name prefix (e.g. "Alice: Hello" -> "Hello")
                for name in ["Alice", "Bob"]:
                    if content.startswith(f"{name}:"):
                        content = content[len(name) + 1:].strip()
                history.append((speaker, content))
                turn_count += 1
                if turn_count % 10 == 0 or turn_count <= 3:
                    print(f"  Turn {turn_count}: {speaker}: {content[:80]}...")

        # Write output
        with open(f"{args.output}_{STARTING_TRANSCRIPT_PATH.split('/')[-1].split('.')[0]}_{topic_name}.txt", "w") as f:
            for name, text in history:
                f.write(f"{name}: {text}\n")

        print("-" * 60)
        print(f"Done! {len(history)} turns")
        print(f"Saved to: {args.output}_{STARTING_TRANSCRIPT_PATH.split('/')[-1].split('.')[0]}_{topic_name}.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a long two-speaker transcript with AutoGen"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model name"
    )
    parser.add_argument(
        "--turns", type=int, default=200,
        help="Number of conversation turns"
    )
    parser.add_argument(
        "--output", default="transcript_long_continue",
        help="Output file path prefix"
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
