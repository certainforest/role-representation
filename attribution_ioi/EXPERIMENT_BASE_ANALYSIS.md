
Experiment Base: Minimal four line set up of "I am Alice" "I am Bob" "I live in France." "I live in Thailand."

Step 0: Process the entire input text, for each entity or attribute, assign it a binding id (think triangle or rectangular) base on the pattern of which line it is on (odd or even lines). (Alice and France gets triangle, Bob and Thailand gets rectangular.)
Step 1: Read the query entity("Alice") or attribute ("France")
Step 2: Find prior occurance of the query, get it's binding id (triangle)
Step 3: Use the binding id as the lookup key to find the country or entity that has it.
Step 4: Copy the answer to final token ":"

type: 
- Swap means swapping "Alice"/"Bob", "France"/"Thailand". So the answer to the base prompt flips.
- Null means replacing "Alice" with "Claire", "France" with "Germany". So the answer to the base prompt becomes undetermined (None means can be anything)


We can see consistently that spikes at the query token in early-mid layers only show up in BINDING settings, and never in CONTROL settings.

What does this tell us:
Take the Q_name setup, where we do `Query(Name=Alice; Country=?) -> Slot1 -> Slot1.country=France`

Our hypothesis is that the model: we can see that the model performs Step 0 and Step 1 early, Step 2 from layer 5 to 13. Then Step 3 look up at layer 13, then Step 4 copying around layer 15.

Take the NULL setup:
In the Binding setting, @/mnt/ssd/aryawu/role-representation/activation_patching/null_results_llama/A1_names_nq.png, 
1. we can see a strip at "Claire" in 
when we patch the base's first line's "Claire" token with "Alice" in the first ~5 layers. This is because we are changing the triangle binding id's associated name early enough (before layer 5), so later when we ask about Alice, the model resolves the binding id triangle at layer 13, and through triangle it found "Alice".
If we patch it later than layer 5, the model would have done Step1:look at query and tried to do Step2: Get it's binding id. It failed to find any "Alice" from prior context, and failed to retrieve the triangle.

2. We can see a strip at "Alice" in the query line when we patch the base's query "Alice" with the source's query "Alice". Here the source's query token "Alice" carry the information of "Alice<->triangle", so even if in the base prompt, the model have never seen "Alice" before, it still resolve the binding id to be triangle, and then lookedup the answer "France" at layer 13
If we patch it before layer 5, the model have not started to process the query yet, so it would not work.
When we patch before layer 13, the model have not fully resolved the binding id to be triangle yet, so it can still take in the information about the lookup key being the traingle, and get France at layer 13. 
If we patch after layer 13, the model have already decided its answer and copied to final token, so it is too late.


In the Control setting, @/mnt/ssd/aryawu/role-representation/activation_patching/null_results_llama/B2_country_nq_ctrl.png.
The model sees Alice from the very start, and can resolve to triangle without any need of patching. The patching only changed the country associated with triangle to go from "Germany" to "France". 

Most important: The fact that we do not observe a strip at the query token "Alice", shows that that token does not carry any information about "France". i.e, the model does not associate Alice<->France. Instead, the model track some intermediate thing, the binding id. In the control setting, we did not change "Alice" to "Claire", so the model's knowledge of Alice's binding id, i.e the triangle, remains.

we can see a single long stript at the "Germany" when patched with "France". This suggests the model does the lookup using triangle in late layers ~28, so any patching at "Germany" with "France" up till layer 28 would allow the model to lookup "France".

The patching is useful up until layer 28, but in the Binding case we had assumed that at layer 15 the model already moved the answer to the end. This is because our BASE differed. When the binding id is unambiguous, the model resolves it very fast. When the queried name/attribute does not exist, the model takes a few more layers to find out.

What about the Alice swap with Bob setting? Originally the model would've had quickly resolved the binding id key to be rectangle, and find Thailand.  
We see that patching in layer 1 and 2 of the line 1 "Bob" to "Alice" works a little. The model now sees two Alice, so its confusing, and it tend to answer the first country it see due to bias.
Patching in layer 3-5 of the second "Alice" to "Bob" makes it see no Alice, so again confused.
Patching "Alice" in query at layer 9-12 has clear effect, this is when we patched in the information of Alice->triangle, so then the model can use it to find "France".


Through head voting we find head L13H18 and L15H8 to be only fired for binding task, and not for control task.
We hypothesis:
L13H18 does the lookup. It reads from residual stream/MLP that the binding id to use as the lookup key should be triangle, and it gets France.
L15H8 moves that information ?