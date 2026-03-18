**Mean Interventions on LLaMA-3.1-8B-Instruct.**

Feng & Steinhardt (2024) hypothesize that binding IDs are vectors additively attached to entity and attribute tokens in the residual stream. To extract the binding ID difference vector ∆E, we run the model on two versions of the same context. In the default ordering, the prompt reads *"I am Alice. I am Bob. I live in France. I live in Thailand."* and we record the residual stream at Bob's token. In the shifted ordering, *"I am Bob. I am Alice. I live in Thailand. I live in France."*, Bob is now in slot-0 and we record his residual stream again. The difference — Bob's representation at slot-1 minus Bob's representation at slot-0 — isolates the binding ID signal, stripping away Bob's identity and leaving only the "I am slot-1" marker. Averaging this difference across 500 name/country pairs gives ∆E. We estimate ∆A identically using the country tokens. We then test: if we add ∆E to Alice's token and subtract it from Bob's, does the model now answer "Thailand" when asked where Alice lives?

We evaluate on 100 held-out samples, extending the original evaluation to include country queries ("Who lives in France?") alongside entity queries ("Where does Alice live?").

| | Control | Attribute | Entity | Both | Random |
|---|---|---|---|---|---|
| Querying E0 | 100% | 0% | 0% | 100% | 100% |
| Querying E1 | 100% | 0% | 0% | 100% | 99% |
| Querying A0 | 100% | 0% | 0% | 79% | 99% |
| Querying A1 | 100% | 1% | 0% | 100% | 100% |

*Table 1: Mean calibrated accuracies for mean interventions on LLaMA-3.1-8B-Instruct.*

Results match the paper's predictions cleanly for entity queries. The Both condition shows slightly weaker cancellation for slot-0 country queries (79%), suggesting a mild asymmetry in how binding IDs are read out when querying by country versus by name.