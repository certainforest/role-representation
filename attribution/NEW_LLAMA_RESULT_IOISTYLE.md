# Llama-3.1-8B-Instruct: IOI-Style Circuit Analysis Results

## Experiment Setup
8 experiments: 2 examples (France/Thailand, Basketball/Soccer) × 4 conditions:
- 1.1 Query_Entity_Swap_Entity (BINDING)
- 1.2 Query_Attr_Swap_Attr (BINDING)
- 1.3 Query_Entity_Swap_Attr (CONTROL)
- 1.4 Query_Attr_Swap_Entity (CONTROL)

---

## Top-10 Heads Per Experiment

| Experiment | Kind | Top 5 Heads |
|---|---|---|
| ex1_1_1 | BINDING | (13,18), (10,1), (15,8), (14,7), (15,11) |
| ex1_1_2 | BINDING | (13,18), (15,8), (17,24), (15,11), (18,23) |
| ex1_1_3 | CONTROL | (17,24), (27,20), (24,3), (17,23), (31,7) |
| ex1_1_4 | CONTROL | (14,4), (16,1), (24,27), (14,22), (17,24) |
| ex2_1_1 | BINDING | (13,18), (15,8), (17,24), (15,11), (16,1) |
| ex2_1_2 | BINDING | (13,18), (15,8), (17,24), (15,11), (18,23) |
| ex2_1_3 | CONTROL | (17,24), (16,1), (27,20), (23,5), (16,22) |
| ex2_1_4 | CONTROL | (17,24), (16,22), (26,12), (18,23), (27,20) |

---

## Head Classification (top-5 voting)

Sets used:
- `binding_set` = union of top-5 across all 4 BINDING experiments
- `control_set` = union of top-5 across all 4 CONTROL experiments
- `entity_set`  = union of top-5 across all 4 entity-swap experiments (1.1 + 1.4)
- `attr_set`    = union of top-5 across all 4 attr-swap experiments (1.2 + 1.3)

**Binding-only** (`binding_set − control_set`):
`(13,18), (10,1), (15,8), (14,7), (15,11)`

**Control-only** (`control_set − binding_set`):
`(27,20), (24,3), (17,23), (31,7), (14,4), (24,27), (14,22), (23,5), (16,22), (26,12)`

**Shared binding+control** (`binding_set ∩ control_set`):
`(17,24), (18,23), (16,1)`

---

## Swap-Type Specificity (top-5 voting)

**Entity-swap-only** (`entity_set − attr_set`):
`(10,1), (14,7), (14,4), (24,27), (14,22), (26,12)`

**Attr-swap-only** (`attr_set − entity_set`):
`(24,3), (17,23), (31,7), (23,5)`

**Shared entity+attr** (`entity_set ∩ attr_set`):
`(13,18), (15,8), (15,11), (17,24), (18,23), (16,1), (16,22), (27,20)`

---

## Key Findings

### Binding-ID Heads: (13,18), (15,8), (15,11)
- All three appear in **all 4 binding experiments** top-5, never in any control top-5.
- All three are in **both entity-swap and attr-swap** conditions → agnostic to swap type.
- (10,1) and (14,7) are also binding-only but entity-swap-specific (name order encoding).

### General Task Heads (shared binding+control): (17,24), (18,23), (16,1)
- Appear in both binding and control top-5 → not specific to binding-ID resolution.
- Likely perform general answer-copying to END token.
- Note: (27,20) is **control-only** from top-5 data, not a general task head.

### Swap-Type Specificity
- **Entity-swap-only:** (10,1), (14,7) — encoding name introduction order.
- **Attr-swap-only:** (24,3), (17,23), (31,7), (23,5) — all control-only, likely encoding attribute order for general lookup.
- (13,18), (15,8), (15,11) are shared across both swap types → true binding-ID heads.

---

## Information Flow Summary

- **Layer 13–15:** L13H18, L15H8, L15H11 resolve binding ID — identify which slot the queried entity/attribute occupies. Fire on binding conditions regardless of swap type.
- **Layers 16–18:** Heads (17,24), (18,23), (16,1) copy resolved slot info toward END (shared with control, so general output heads).