# Binding Head Analysis: Llama-3.1-8B vs Qwen3-8B

## Experiment: ex1_1_1
- SOURCE: Alice first → "I live in France" → Answer: France
- BASE: Bob first → "I live in France" → Answer: Thailand
- diff_pos = [11, 15] = [' Alice', ' Bob'] (swapped introduction order)
- query Alice = pos 31 (in "Where does **Alice** live?")
- intro Alice = pos 11 (in "I am **Alice**.")

---

## Llama-3.1-8B — Binding Head: L13H18

### Phase 0: Which component carries the signal?
```
q       : 0.676   ← dominant
k       : 0.040
v       : -0.009
pattern : 0.719   ← driven by Q (consistent)
z       : 0.859
```
**Q-circuit dominant.** The head resolves binding by deciding *where to attend* (Q×K), not what to copy (V).

### Attention Pattern of L13H18
- **SOURCE** (Alice first): final token ':' attends to `'."\n'`(0.31), `'?'`(0.12), `' Alice'`(0.10)
- **BASE** (Bob first): same query token attends differently — shifts toward Bob-related positions
- The head reads the **dialogue line-ending tokens** (`'."\n'`) to determine which turn Alice spoke on.
- Q at pos 31 (query Alice): SOURCE attends to `'."\n'`(0.22) and `' Alice'`(0.12); BASE attends to `'."\n'`(0.41) and `' Bob'`(0.20) and `' Thailand'`(0.14) — the head knows which name is in which slot.

### Residual Decomposition: what writes into L13H18's Q-space (at pos 35)?
Top heads by contribution to binding direction in Q-space:
```
L12H2:  +0.188   L12H4:  +0.120   L12H15: +0.104
L12H7:  -0.103   L11H25: -0.099   L12H13: +0.096
L10H1:  +0.081   L12H25: +0.064
```
Top MLP contributions:
```
L12: +0.149   L11: +0.027   L10: +0.018
```
**L12 (both heads and MLP) is the critical preparation layer** immediately before L13H18. L10H1 is notable (also a binding-only head from voting).

### Trace: what writes to query Alice (pos 31)?
Top heads by ||src_output − base_output|| at pos 31:
```
L13H18: 1.630   ← the binding head itself writes most to query Alice
L14H19: 1.347   SOURCE attends to intro Alice(0.89); BASE attends to intro Bob(0.94)
L12H12: 1.297   attends to Alice in both — same attention, different value
L15H10: 1.256
L13H16: 1.102
```
First layer with above-threshold head: **L5H9**

Top MLP layers by diff at query Alice:
```
L31: 5.28  L14: 3.00  L13: 2.78  L12: 2.63  L11: 2.48
```
(L31 is downstream of L13H18, so its large value is a consequence, not a cause.)

---

## Qwen3-8B — Binding Head: L23H26

### Phase 0: Which component carries the signal?
```
q       : 0.039
k       : 0.000
v       : 0.490   ← dominant
pattern : 0.020
z       : 0.629
```
**V-circuit dominant.** The head already attends to the right place (query Alice); what differs is *what it reads from there* (V).

### Attention Pattern of L23H26
- Final token ':' attends to **`' Alice'` (pos 31)** with weight 0.32 in SOURCE
- Attention pattern is essentially the same between SOURCE and BASE — the head consistently points to query Alice
- The binding signal is in the V vector at Alice's position, not in the attention routing

### Residual Decomposition: what writes into L23H26's V-space (at pos 31)?
Top heads by contribution to binding direction in V-space:
```
L18H14: -0.275   L17H7:  -0.274   L18H15: +0.248
L15H9:  +0.195   L15H10: -0.189   L19H24: +0.154
L21H10: +0.153   L13H19: +0.150   L15H11: +0.144
```
Top MLP contributions:
```
L22: +0.732   L20: +0.601   L21: +0.279   L18: +0.285   L15: +0.301
L12: -0.387   L17: -0.200   L16: -0.150
```
**MLPs L20–L22 are the dominant writers** into L23H26's V-space at query Alice — the binding representation is primarily built by MLPs in the layers just before the binding head. L15H9 and L19H24 are notable attention heads (L15H9 appeared in binding-only voting).

### Trace: what writes to query Alice (pos 31)?
Top heads by ||src_output − base_output|| at pos 31:
```
L28H21: 15.157   SOURCE: intro Alice(0.53); BASE: intro Bob(0.56)  ← slot-copying head
L34H7:  10.013
L22H8:   8.779   SOURCE: intro Alice(0.69); BASE: intro Alice(0.65)
L16H22:  8.524   SOURCE: Thailand/Bob/line-endings; BASE: line-endings/live
L19H24:  8.016   SOURCE: intro Alice(0.74); BASE: intro Bob(0.82)
```
First layer with above-threshold head: **L11H11**

Top MLP layers by diff at query Alice:
```
L35: 39.16  L34: 37.77  L33: 38.17  L32: 32.48  L31: 26.96
```
Note: these large MLP diffs are at layers 31–35, which are **after** L23H26 (layer 23). They reflect downstream consequences, not causes. The causal MLP layers are 15–22.

Key attention head finding: **L19H24** and **L28H21** both show the canonical pattern —
SOURCE attends to intro Alice, BASE attends to intro Bob. These are the heads that move binding info from the introduction token to the query token position.

---

## Overall Understanding

### The Binding Circuit (both models)

```
[Early layers 0-10]
  Intro Alice (pos 11) and intro Bob (pos 15) tokens
  get positional/contextual encoding from early heads.
  First binding signal appears ~L5 (Llama) / ~L11 (Qwen).

[Mid layers — INFORMATION MOVEMENT]
  Key heads attend from query Alice (pos 31)
  → intro Alice/Bob (pos 11/15) depending on who was introduced first.
  This MOVES the slot-identity of Alice to the query position.
    Llama: L13H18, L14H19, L12H12 (layers 12-15)
    Qwen:  L19H24, L22H8, L28H21  (layers 19-28)
  MLPs in these layers further consolidate the binding representation.

[Binding-ID head]
  Llama L13H18: Q-dominant — uses the slot info at pos 35 (last token)
                to route attention to the correct attribute position
  Qwen  L23H26: V-dominant — consistently attends to query Alice (pos 31)
                and reads the pre-computed binding signal via V

[Output heads — copy to END]
  Llama: L17H24, L18H23
  Qwen:  L28H0, L29H11
```

### Mechanistic Difference: Llama vs Qwen

| | Llama L13H18 | Qwen L23H26 |
|---|---|---|
| Dominant component | Q (0.68) | V (0.49) |
| Head role | **Resolver** — computes slot via Q×K | **Reader** — extracts pre-computed slot via V |
| Q/V reading position | Last token ':' (pos 35) | Query Alice (pos 31) |
| Attention pattern changes? | Yes — attends differently SOURCE vs BASE | No — always attends to query Alice |
| Binding info built by | L12 heads + L12 MLP | MLPs L15–L22 |
| First signal appears | L5H9 | L11H11 |

Both implement the same abstract computation — map (query entity name) → (slot index) → (attribute value) — but through different circuit implementations.

---

## Trace: what writes to intro Alice (pos 11)? (trace_query_alice.py --target intro_alice)

### Llama top heads at pos 11:
```
L31H23: 5.270   L31H14: 3.676   L30H25: 1.714   L30H27: 1.652   L16H22: 1.520
```
First above-threshold head: L9H3. Top MLP layers: L31(19.60), L30(9.15), L17(7.54)

### Qwen top heads at pos 11:
```
L35H24: 29.022   L35H28: 25.543   L33H8: 22.521   L35H25: 21.349   L34H28: 20.860
```
First above-threshold head: L23H6. Top MLP layers: L34(107.77), L35(103.75), L33(99.77)

**Key finding**: The large diffs at intro Alice are at LATE layers (30–31 for Llama, 33–35 for Qwen) — all **downstream** of the binding heads (L13/L23). The late-layer heads writing to intro Alice are *consequences* of the binding resolution, not causes. The early causal encoding (L9H3 Llama, L23H6 Qwen) is subtle and weak.

---

## Info-Mover Verification (verify_info_movers.py)

### Llama L14H19:
```
full=0.013  at_query_alice=0.013  at_intro_alice=0.000  at_diff_pos=0.004
```

### Qwen L19H24:
```
full=0.059  at_query_alice=0.079  at_intro_alice=0.000  at_diff_pos=0.020
```

### Qwen L28H21:
```
full=0.020  at_query_alice=0.000  at_intro_alice=0.000  at_diff_pos=0.000
```

**Key finding**: All candidate info-moving heads have very low causal patch metrics (0.01–0.08). Despite having large output diffs (high ||src−base||), patching them individually doesn't restore the binding metric. The binding circuit is **highly distributed** — no single head is a causal bottleneck. Information flows through many parallel paths simultaneously.

---

## Fundamental Limitation of the Experimental Setup

**Critical insight**: In SOURCE vs BASE, **binding exists in BOTH conditions** — Alice is bound to slot 1 in SOURCE, slot 2 in BASE. The model performs the binding operation identically in both; only the slot assignment differs.

Therefore our patching measures: *"which components encode Alice's slot identity (slot 1 vs slot 2)"*

It **cannot** find: *"which components compute the binding operation itself"* — because those components fire equally in both conditions and are invisible to our metric.

This explains why:
- Info-moving heads show large output diffs but low causal metrics (slot encoding is redundant/distributed)
- No single bottleneck head found — the circuit looks distributed because we're measuring something redundantly encoded
- The circuit analysis hits a wall at a certain depth

**To find where binding is actually computed, need a NULL condition where binding is absent:**
- **Option 1 (cleanest)**: Remove name-introduction lines entirely → `"I live in France.\nI live in Thailand.\nQuestion: Where does Alice live?"` — Alice never introduced, no binding possible
- **Option 2**: Replace names with anonymous tokens [X], [Y] — structure preserved but unresolvable. Or random names eg. Claire, David.
- **Option 3**: Query a name not in the transcript ("Where does Carol live?")
- **Option 4**: Repeat same name ("I am Alice.\nI am Alice.\n...") — ambiguous binding

Option 1 is cleanest: SOURCE (binding) vs NULL (no binding). Components that fire in SOURCE but not NULL are the actual binding-computation components.

---

## What We Still Don't Know

1. **Where is binding actually computed?** The current setup can't answer this — needs NULL redesign.
2. **What is L12 MLP doing in Llama?** Biggest single contributor to L13H18's Q-space, but MLP internals are opaque.
3. **Does the circuit generalize across names?** All experiments used Alice/Bob. Would same heads fire for Carol/Dave?
