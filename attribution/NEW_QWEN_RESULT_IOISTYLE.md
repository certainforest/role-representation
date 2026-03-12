# Qwen/Qwen3-8B: IOI-Style Circuit Analysis Results

## Experiment Setup
8 experiments: 2 examples (France/Thailand, Basketball/Soccer) × 4 conditions:
- 1.1 Query_Entity_Swap_Entity (BINDING)
- 1.2 Query_Attr_Swap_Attr (BINDING)
- 1.3 Query_Entity_Swap_Attr (CONTROL)
- 1.4 Query_Attr_Swap_Entity (CONTROL)

---

## Top-5 Heads Per Experiment

| Experiment | Kind | Top 5 Heads |
|---|---|---|
| ex1_1_1 | BINDING  | (23,26), (15,9), (21,18), (26,26), (24,31) |
| ex1_1_2 | BINDING  | (23,26), (28,0), (29,11), (21,19), (30,7) |
| ex1_1_3 | CONTROL  | (31,3), (24,26), (26,26), (23,6), (34,0) |
| ex1_1_4 | CONTROL  | (23,6), (28,0), (29,11), (24,29), (29,6) |
| ex2_1_1 | BINDING  | (23,26), (21,18), (22,10), (24,23), (18,15) |
| ex2_1_2 | BINDING  | (23,26), (28,0), (29,11), (30,5), (30,7) |
| ex2_1_3 | CONTROL  | (24,29), (23,6), (29,11), (24,23), (30,7) |
| ex2_1_4 | CONTROL  | (23,6), (28,0), (24,29), (29,11), (30,5) |

---

## Head Classification (top-5 voting)

Sets used:
- `binding_set` = union of top-5 across all 4 BINDING experiments
- `control_set` = union of top-5 across all 4 CONTROL experiments
- `entity_set`  = union of top-5 across all 4 entity-swap experiments (1.1 + 1.4)
- `attr_set`    = union of top-5 across all 4 attr-swap experiments (1.2 + 1.3)

**Binding-only** (`binding_set − control_set`):
`(23,26), (15,9), (21,18), (24,31), (21,19), (22,10), (18,15)`

**Control-only** (`control_set − binding_set`):
`(31,3), (24,26), (23,6), (34,0), (24,29), (29,6)`

**Shared binding+control** (`binding_set ∩ control_set`):
`(26,26), (28,0), (29,11), (30,7), (24,23), (30,5)`

---

## Swap-Type Specificity (top-5 voting)

**Entity-swap-only** (`entity_set − attr_set`):
`(15,9), (21,18), (24,31), (29,6), (22,10), (18,15)`

**Attr-swap-only** (`attr_set − entity_set`):
`(21,19), (30,7), (31,3), (24,26), (34,0)`

**Shared entity+attr** (`entity_set ∩ attr_set`):
`(23,26), (26,26), (23,6), (28,0), (29,11), (24,29), (24,23), (30,5)`

---

## Key Findings

### The Binding-ID Head: (23,26)
- Rank **#1 in ALL 4 binding experiments**, never appears in any control top-5.
- Shared across **both entity-swap and attr-swap** → agnostic to what was swapped.
- Direct analog of Llama's **(13,18)**.

### Other Binding-Only Heads
- **(21,18), (22,10), (18,15)** — entity-swap-only → encode name introduction order.
- **(21,19)** — attr-swap-only → encodes attribute order.
- **(24,31)** — entity-swap-only, appears only in ex1 (France/Thailand).

### General Task Heads (shared binding+control)
- `(26,26), (28,0), (29,11), (30,7), (24,23), (30,5)` — needed for both binding and control.
- Likely perform general answer lookup/copy to END token.
- Note: (28,0) and (29,11) appear heavily in control conditions — more general than binding-specific.

---

## Comparison with Llama

| Property | Llama-3.1-8B | Qwen3-8B |
|---|---|---|
| Primary binding-ID head | L13H18 | L23H26 |
| Secondary binding heads | L15H8, L15H11 | L21H18, L22H10 |
| General task heads | L17H24, L18H23, L16H1 | L28H0, L29H11, L30H7 |
| Binding-ID head layer | ~13 (early-mid) | ~23 (mid-late) |
| Binding-ID head shared e+a | Yes | Yes |
