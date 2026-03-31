# Binding-ID Counter: Findings

## Scripts and Outputs

| Script | Output | Description |
|--------|--------|-------------|
| `binding_id_summary_plot.py` | `binding_id_summary.png` | Original summary: onset curves + context sensitivity bar chart. Hi probe trained on a0 only (truncated prompt). |
| `binding_id_summary_plot_new.py` | `binding_id_summary_new.png` | Updated version: full hi prompt ("What about you?" + a1 line), probe trained on both a0 and a1. |
| `counter_token_heatmap.py` | `binding_id_token_heatmap.png` | Token-level heatmap: P(△) at every token × every layer, averaged over 20 examples. OOD tokens greyed out. |

---

## Hypothesis Being Tested

The model tracks speaker identity using abstract **binding-ID** slots (△ = intro-first entity, □ = intro-second entity). These are not just positional shortcuts or name-identity associations — they are context-sensitive, abstract representations that:

1. Get encoded progressively as the model processes more context
2. Can be overridden by pre-utterance cues (greetings, address, nomination, narration)
3. Are carried by semantically meaningful token positions, not spread uniformly across the sentence

---

## Key Findings

### 1. Progressive encoding across token types (`binding_id_summary_new.png`, Panel 1)

Binding-ID appears at different layers depending on how much context is needed to resolve it:

| Token | Onset | Why |
|-------|-------|-----|
| Entity name (`"I am Alice"`) | L0 | Intro order is immediately available from embedding |
| `"I live"` token | L3 | Zero-shot — probe never trained here. Signal propagates via attention from entity tokens (~3 layers of lag) |
| Country token, Hi setup | L6 | Greeting must be resolved first, then routed to country token — more computation required |
| Country token, Base setup | L0 | No greeting ambiguity; intro order directly determines speaker |

**Interpretation:** The latency difference (L0 → L3 → L6) reflects the amount of contextual computation the model must do before it can assign binding-ID at each position. The "I live" token is particularly notable: the probe was never trained on pronouns, yet it detects binding-ID there from L3 — the model propagates the speaker's identity to the first-person pronoun in the utterance.

Note: the original `binding_id_summary.png` reported Hi country onset at L6 because the probe was trained only on a0 (truncated prompt). The updated `binding_id_summary_new.png` trains on both a0 and a1 using the full prompt, giving the more honest estimate of L8.

### 2. Pre-utterance anchors all correctly assign □; Hi e0! uniquely disturbs to △ (`binding_id_summary_new.png`, Panel 2)

At L13, P(△) at the country token:

| Context | P(△) | Interpretation |
|---------|-------|---------------|
| Base | ~0.90 | Correct: e0=△ speaks first country |
| Hi e0! (confused) | ~0.97 | Disturbed: linguistically anomalous greeting causes e0=△ to speak, preserving intro order |
| Hi e1! | ~0.04 | Correct: e1=□ greeted → e1 speaks → country gets □ |
| Address e1 | ~0.04 | Correct anchor |
| Nomination e1 | ~0.04 | Correct anchor |
| Narration "said" | ~0.04 | Correct anchor |
| Retrospective (post-hoc) | ~0.88 | Model ignores this; binding-ID unchanged |

**Interpretation:** The model reliably uses pre-utterance cues to assign binding-ID before the utterance begins. The one exception is "Hi e0!" (where e0 greets themselves — linguistically anomalous), which causes the assignment to default back to intro order (△). This is the "reset/disturb" phenomenon. It is not a true reset of the mechanism; it is an anomalous input causing the greeting-override to fail.

### 3. Binding-ID is keyword-local, not sentence-distributed (`binding_id_token_heatmap.png`)

The token-level heatmap (averaged over 20 examples, OOD tokens greyed out) shows:

- **Entity name tokens** (e.g., "Alice", "Bob"): strong binding-ID from ~L5, stable across all setups — reflects intro order and never changes
- **"I" tokens** (in "I am e0", "I am e1"): mirror the entity name tokens — the first-person pronoun in the intro line carries the same binding-ID as the name it follows
- **"I live" token and country token** (a0, a1): these **flip between setups** — red (△) in base and hi-confused, blue (□) in hi-e1. This is the key contrast showing the greeting overrides the binding-ID at the utterance level
- **All other tokens** (greyed): no interpretable signal — binding-ID is not spread across function words, punctuation, or filler tokens

**Interpretation:** The binding-ID representation is localized to semantically meaningful positions. The model does not broadcast this information uniformly; it concentrates at entity anchors (name tokens) and attribute anchors (country tokens and their speaker's "I"). This is consistent with a targeted lookup mechanism rather than a diffuse representation.

### 4. Zero-shot generalization

The probe was trained only on entity name tokens and a0 country tokens. Despite this, it correctly recovers binding-ID at:

- **"I live" tokens** (zero-shot): reaches near-1.0 by L13, confirming propagation from entity tokens
- **a1 country token** (zero-shot before adding to training): also correct — the second country token picks up the binding-ID of whoever responds to "What about you?"

This confirms the binding-ID direction is a **shared linear subspace** across different token types within the same dialogue context — not specific to the positions the probe was trained on.

---

## Does This Verify the Hypothesis?

**Yes, with nuance.**

✅ Binding-IDs are abstract: the 4-cond crossed design eliminates name-identity and positional confounds. The probe generalizes across names (name-split = 1.0).

✅ They are progressively encoded: L0 (entity) → L3 (speaker pronoun) → L8 (attributed country). The lag reflects computational cost, not arbitrary choice.

✅ Greeting overrides intro order: hi-e1 and hi-confused produce opposite binding-IDs at the country token despite identical intro sequences.

✅ Pre-utterance anchors work: address, nomination, narration all correctly assign □ to the addressed/nominated/mentioned party's country token.

✅ Hi e0! (confused) is the anomalous disruptor: the one context that breaks normal assignment.

✅ Binding-ID is keyword-local: concentrated at entity and attribute tokens, not diffuse.

**Nuance / open questions:**

- The "nothing hypothesis" was confirmed for Hi country at early layers (L0–L5, P(△) ≈ 0.5) — the model does not default to intro order and then flip. It waits until it has resolved the greeting before assigning binding-ID.
- Late layers (L20+) are unreliable: probe train accuracy declines, P(△) saturates to 1.0 for all templates — likely overfitting to a spurious feature unrelated to binding-ID.
- The base country early onset (L0) is real and meaningful: when there is no linguistic cue (no greeting), the model assigns binding-ID by sequential order immediately — first speaker = △. The contrast with Hi (onset L8) shows that sequential order is the default, and the model only overrides it when a stronger linguistic cue is present.