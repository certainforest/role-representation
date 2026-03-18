Experiment Hi: five line set up of "I am Alice" "I am Bob" "Hi Bob!" "I live in Thailand. Where do you live?" "I live in France."

In the previous experiment base, introduction order and line parity coincide, names appear in fixed alternating positions and attributes follow the same pattern. Because these two positional signals are perfectly correlated, the model can rely on a simple positional shortcut rather than constructing an explicit entity–attribute mapping.

The above prompt removes this coincidence. By inserting dialogue turns, the line ordering no longer corresponds directly to entity identity. To answer correctly, the model must understand the dialogue structure where the first speaker speaks odd lines, and the second speaker speaks even lines. The model can then solve the task using a line-parity binding heuristic.

Same hypothesis of model's algorithm as before:
Step 0: Process the entire input text, for each entity or attribute, assign it a binding id (think triangle or rectangular) base on the pattern of which line it is on (odd or even lines). (Alice and France gets triangle, Bob and Thailand gets rectangular.)
Step 1: Read the query entity("Alice") or attribute ("France")
Step 2: Find prior occurance of the query, get it's binding id (triangle)
Step 3: Use the binding id as the lookup key to find the country or entity that has it.
Step 4: Copy the answer to final token ":"
Note we consider the triangle as line parity, which might or might not imply "spoke by alice", depending on if the model constructed a alice speaker representation. We do not care about the binding id's exact meaning for now, only that it tracks something.

type: 
- Swap means swapping "Alice"/"Bob", "France"/"Thailand". So the answer to the base prompt flips.
- Null means replacing "Alice" with "Claire", "France" with "Germany". So the answer to the base prompt becomes undetermined (None means can be anything)


We can still see consistently the exact same spike pattern comparing to each setup's counterpart from Experiment base.

Here it is more important to look at the Q_country setup, because we've broken the country attribute's appearance ordering. In Q_country we do `Query(Country=France; Name=?) -> Slot1(Line parity odd) -> Slot1.name=Alice`

Our hypothesis is still that the model: performs Step 0 and Step 1 early, Step 2 from layer 5 to 10. Then Step 3 look up at layer 13, then Step 4 copying at layer 15. We get the additional information that step 4 might actually happens at the question mark token.

Take the NULL setup:
In the binding setting @/mnt/ssd/aryawu/role-representation/activation_patching/hi_results_llama/Q_country_hi_bind.png
1. We see a strip at "Germany" patched by "France" in the last line "I live in Germany" spoke by Alice. This line is the country line that have the binding id of triangle. So when we change the triangle binding id's associated country early enough, before layer 5 where Step 2 happens, the model can look up and see "France" later.

2. We see a strip at "France" in the query, when patched with the "France" from source that have seen Alice say "I live in France", which carries the encoding of France<->triangle. This allows the model in base prompt to resolve the binding id to triangle, and then lookup Alice as the answer. 
The strip at "France" got shorter, only layer 5 - 10. We additioanlly see new strips at the quesiton mark. This suggests the lookup happens at the end of the query, on around layer 13-15. Then the final answer is copied to last token around layer 15.

In the Control setting, we see the exact same thing as the experiment base.

The fact that we observe the same thing as Experiment Base suggests that the model is not tracking the ordering of attributes, but actually associating binding id to each line, following the transcript conversation's setup. 
By doing simple inference experiments we can see that the model relies on linguistic/conversational hints like "hi bob!" or "What about you?" in order to do the task right, suggesting that these hints tells the model to assign binding ids per line, or i.e. make binding id to have speaker tracking functionality.

Considering the SWAP setup in Experiment Hi Q_country:
1. We see a white strip at when Bob's country get patched from "France" to be "Thailand". The model now can't see France at all, so gets confused.
We additionally see a red square at layer 2 of the '".' in the middle of Bob's sentence "I live in Thailand. What about you?" from Source. This means that token carried information about Thailand associated with the binding id of rectangle, and thus when asked about France, the model knows to not answer Bob.

2. We see a red strip at Alice's country get patched from "Thailand" to "France". The model now sees two France, it is possible that France's binding id got updated from rectangle to triangle, and thus model answered Alice. It is also possible that model just defaults to Alice.

3. We see the red strip at "France" in query around layer 10-15. Again, here the "France" token carry the binding id = triangle information, and the model answers Alice.


(ignore below)
Here's the same writing but for Q_name setup: where we do `Query(Name=Alice; Country=?) -> Slot1 -> Slot1.country=France`

Our hypothesis is that the model: we can see that the model performs Step 0 and Step 1 early, Step 2 from layer 5 to 10. Then Step 3 look up at layer 13, where the model then copies answer to the final token. We get the additional information that step 3 actually happens at the question mark token.

Take the NULL setup:
In the Binding setting, @/mnt/ssd/aryawu/role-representation/activation_patching/hi_results_llama/Q_name_hi_bind.png, 
1. we can see a strip at "Claire" when we patch the base's first line's "Claire" token with "Alice" in the first ~5 layers, again because we are changing the triangle binding id's associated name early enough (before layer 5), so later when we ask about Alice, the model resolves the binding id triangle at layer 13, and through triangle it found "Alice".
We additionally see red sparks at the ending quotation mark '."\n' after "Claire", suggeseting that each sentence's ending carries the "Alice" and binding id information. 

2. We can also still see a strip at "Alice" in the query line when we patch the base's query "Alice" with the source's query "Alice". Here the source's query token "Alice" carry the information of "Alice<->triangle", so even if in the base prompt, the model have never seen "Alice" before, it still resolve the binding id to be triangle, and then lookedup the answer "France" at layer 13
The strip at "Alice" got shorter, only layer 5 - 10. We additioanlly see new strips at the quesiton mark. This suggests the lookup happens at the end of the query, on around layer 13-15. Then the final answer is copied to last token on layer 15.


In the Control setting, we see the exact same thing as the experiment base.