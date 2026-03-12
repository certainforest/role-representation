# V1
Example1:
1.1 Query_Entity_Swap_Entity

SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I live in France."
"I live in Thailand."
Question: Where does Alice live? Answer:
" France"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Bob."
"I am Alice."
"I live in France."
"I live in Thailand."
Question: Where does Alice live? Answer:
" Thailand"


1.2 Query_Attr_Swap_Attr
SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I live in France."
"I live in Thailand."
Question: Who lives in France? Answer:
" Alice"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I live in Thailand."
"I live in France."
Question: Who lives in France? Answer:
" Bob"


1.3 Query_Entity_Swap_Attr (Controlled)

SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I live in France."
"I live in Thailand."
Question: Where does Alice live? Answer:
" France"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I live in Thailand."
"I live in France."
Question: Where does Alice live? Answer:
" Thailand"


1.4 Query_Attr_Swap_Entity (Controlled)
SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I live in France."
"I live in Thailand."
Question: Who lives in France? Answer:
" Alice"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Bob."
"I am Alice."
"I live in France."
"I live in Thailand."
Question: Who lives in France? Answer:
" Bob"



Example2: 
1.1 Query_Entity_Swap_Entity

SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I like basketball."
"I like soccer."
Question: What sport does Alice like? Answer: Alice likes
" basketball"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Bob."
"I am Alice."
"I like basketball."
"I like soccer."
Question: What sport does Alice like? Answer: Alice likes
" soccer"


1.2 Query_Attr_Swap_Attr
SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I like basketball."
"I like soccer."
Question: Who likes in basketball? Answer:
" Alice"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I like soccer."
"I like basketball."
Question: Who likes in basketball? Answer:
" Bob"


1.3 Query_Entity_Swap_Attr (Controlled)

SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I like basketball."
"I like soccer."
Question: What sport does Alice like? Answer: Alice likes
" basketball"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I like soccer."
"I like basketball."
Question: What sport does Alice like? Answer: Alice likes
" soccer"


1.4 Query_Attr_Swap_Entity (Controlled)
SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I like basketball."
"I like soccer."
Question: Who likes in basketball? Answer:
" Alice"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Bob."
"I am Alice."
"I like basketball."
"I like soccer."
Question: Who likes in basketball? Answer:
" Bob"



NOTES: 
Our assumption/hypothesis is: This is just Binding ID:

It does not matter if we are asking attributes->entity or the reverse.
The spike only shows up if the position of which the attribute/entity showed up changed in the base v.s. source prompt.

For our example 1:
At layer ~20, at the query object token, the model carried information about which positional slot that attribute was in. 
When we patch that activation from SOURCE into BASE, we are transplanting the resolved result: "Alice is in slot 1." The later layers then read that and use it as the lookup key to output France instead of Thailand.

Similarly for example 2: 
At layer ~20, at the query object token, the model carried information about which positional slot that unity was in. 
When we patch that activation from SOURCE into BASE, we are transplanting the resolved result: “basketball is in slot 1.” The later layers then read that and use it as the lookup key to output Alice instead of Bob.

Mechanistic Interpretability Hypothesis:
The model applies a simple binding-ID slot heuristic to associate entities with attributes.

Model Algorithm
1. Identify the queried entity.
2. Determine whether it is the first or second name introduced in the text.
3. Assign a binding ID slot based on this order:
first introduced name → slot #1
second introduced name → slot #2
4. Identify all occurrences of "I live in <Country>".
5. Assign countries binding slots based on their order of appearance:
first country mentioned → slot #1
second country mentioned → slot #2
6. Output the country associated with the same binding slot as the queried name.

Slot(Name) = introduction_order(Name)
Country = country_with_same_slot

================================================
# V2
New examples V2: Breaking the Slot-Ordering Heuristic. Introducing line parity hypothesis.

In the V1 prompts, introduction order and line parity coincide, names appear in fixed alternating positions and attributes follow the same pattern. So both heuristics yield the same slot assignment: introduction_order(Name) == line_parity(Name)
Because these two positional signals are perfectly correlated, the model can rely on a simple positional shortcut rather than constructing an explicit entity–attribute mapping.

The following prompt family removes this coincidence. introduction_order(Name) ≠ line_parity(Name)

By inserting dialogue turns, the line ordering no longer corresponds directly to entity identity. As a result, positional shortcuts based on ordering or parity are no longer sufficient.

To answer correctly, the model must understand the dialogue structure where the first speaker speaks odd lines, and the second speaker speaks even lines. The model can then solve the task using a line-parity binding heuristic.

Example1:
1.1 Query_Entity_Swap_Entity

SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. Where do you live?"
"I live in Thailand. What about you?"
"I live in France."
Question: Where does Alice live? Answer: Alice lives in
" France"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Bob."
"I am Alice."
"Nice to meet you. Where do you live?"
"I live in Thailand. What about you?"
"I live in France."
Question: Where does Alice live? Answer: Alice lives in"
" Thailand"


1.2 Query_Attr_Swap_Attr
SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. Where do you live?"
"I live in Thailand. What about you?"
"I live in France."
Question: Who lives in France? Answer:
" Alice"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. Where do you live?"
"I live in France. What about you?"
"I live in Thailand."
Question: Who lives in France? Answer:
" Bob"


1.3 Query_Entity_Swap_Attr (Controlled)

SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. Where do you live?"
"I live in Thailand. What about you?"
"I live in France."
Question: Where does Alice live? Answer: Alice lives in
" France"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. Where do you live?"
"I live in France. What about you?"
"I live in Thailand."
Question: Where does Alice live? Answer: Alice lives in"
" Thailand"


1.4 Query_Attr_Swap_Entity (Controlled)
SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. Where do you live?"
"I live in Thailand. What about you?"
"I live in France."
Question: Who lives in France? Answer:
" Alice"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Bob."
"I am Alice."
"Nice to meet you. Where do you live?"
"I live in Thailand. What about you?"
"I live in France."
Question: Who lives in France? Answer:
" Bob"



Example2: 
1.1 Query_Entity_Swap_Entity

SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. What sport do you like?"
"I like soccer."
"I like basketball."
Question: What sport does Alice like? Answer: Alice likes
" basketball"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Bob."
"I am Alice."
"Nice to meet you. What sport do you like?"
"I like soccer."
"I like basketball."
Question: What sport does Alice like? Answer: Alice likes
" soccer"


1.2 Query_Attr_Swap_Attr
SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. What sport do you like?"
"I like soccer."
"I like basketball."
Question: Who likes in basketball? Answer:
" Alice"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. What sport do you like?"
"I like basketball."
"I like soccer."
Question: Who likes in basketball? Answer:
" Bob"


1.3 Query_Entity_Swap_Attr (Controlled)

SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. What sport do you like?"
"I like soccer."
"I like basketball."
Question: What sport does Alice like? Answer: Alice likes
" basketball"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. What sport do you like?"
"I like basketball."
"I like soccer."
Question: What sport does Alice like? Answer: Alice likes
" soccer"


1.4 Query_Attr_Swap_Entity (Controlled)
SOURCE (activations taken FROM this prompt):
This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"Nice to meet you. What sport do you like?"
"I like soccer."
"I like basketball."
Question: Who likes in basketball? Answer:
" Alice"

BASE (activations patched INTO this prompt):
This is the transcript of a conversation.
"I am Bob."
"I am Alice."
"Nice to meet you. What sport do you like?"
"I like soccer."
"I like basketball."
Question: Who likes in basketball? Answer:
" Bob"



Here we still see the same spike around layer 20 for the cases needing binding id resolution but not for the controlled ones.

Hypothesis
Mechanistic Interpretability Hypothesis
The model exploits a regular positional pattern in the prompt: names and attributes appear in alternating lines. The model uses line parity as a binding slot.

Model Algorithm
1. Identify the queried entity.
2. Find the line containing "I am <Name>".
3. Determine which line parity slot this line corresponds to.
odd line(first line) → slot #odd
even line(second line) → slot #even
4. Identify the attribute statements "I live in <Country>".
5. Determine their parity slots.
5. Output that country whose slot matches the queried entity.
i.e.
Slot(Name) = line_parity("I am Name")
Country = attribute_with_same_slot