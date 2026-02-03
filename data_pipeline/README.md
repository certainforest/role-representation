We want: A discourse where speaker identity must be inferred pragmatically, not given by tags — and where beliefs must remain separated by inferred agent.
We also need a shared question under discussion (QUD) that the discourse revolves around.

Each data item should just be one turn. There's should not be the case where a new speaker self-selects and starts a new turn. The current speaker should always select the next speaker and transfers the turn to them, otherwise this data item ends.

There's multiple things each item need to satisfy to make it solvable:
1. We need structural speaker disambiguation via discourse logic.
Core idea (Conversation Analysis (Sacks, Schegloff, Jefferson)): Turn-taking is governed by constraints that make certain speaker sequences impossible. So identity is inferred by what would be pragmatically illegal.
eg. repeated question ⇒ new respondent
eg. disagreement/two answers ⇒ new respondent

2. We need role assignments to avoid ambiguity with unlabeled item.
eg. If there is only one interviewer and two intervewee, the questions must belong to the interviewer. Or if there's two interviewer and one interviewee, all answers belong to the interviewee.

3. We need some occasional linguistic clues of names being mentioned.
eg. Questions being asked towards a specific name.
eg. The speaker refering to another speaker's name, essentially selects the next speaker and transfers the turn to them.


Possible templates to the LLM? 
eg. 
repeated question ⇒ new respondent
1. Repeated question allocation (you found this one)
Why did the project fail?
It failed because of funding cuts.
Why did the project fail?
It was mostly due to leadership turnover.

2. Disagreement cannot be self-directed
Why didn’t the policy work?
It failed because costs increased.
I don’t agree — the bigger issue was uncertainty.

3. Clarification requests force role switch
When you say demand slowed, do you mean fewer customers or...?

4. Attribution references
But as you mentioned earlier ..


5. Answering two alternatives
Was the failure due to poor management or market conditions?
Mostly market conditions.
I’d say management played the larger role.
Two incompatible answers → different speakers.


We also need roles.
Otherwise this would be ambiguous. You should assign rolles to them. i.e. make it clear one is the interviewer and the other two is intervewee, thus questions belong tot he interviewer. Or if there's two interviewer and one interviewee, thus all answers belong to the interviewee.

The end of a  Turn-Constructional Units(TCU) is called a Transition Relevance Place (TRP). It’s the spot where a change of speaker becomes relevant—and possible.


Literature: 
https://en.wikipedia.org/wiki/Turn-taking?utm_source=chatgpt.com
Turn-taking structure within a conversation has three components:[8]

The turn-taking component contains the main content of the utterance and is built from various unit types (turn construction units, or TCUs). The end of a TCU is a point where the turn may end and a new speaker may begin, known as a transition relevance place or TRP.
The turn allocation component comprises techniques that select the next speaker. There are two types of techniques: those where the current speaker selects the next speaker, and those where the next speaker selects themself.
Rules govern turn construction and give options to designate the next turn-taker in such a way as to minimize gaps and overlap. Once a TRP is reached, the following rules are applied in order:
The current speaker selects the next speaker and transfers the turn to them; or
One of the non-speakers self-selects, with the first person to speak claiming the next turn; or
No one self-selects, and the current speaker continues until the next TRP or the conversation ends