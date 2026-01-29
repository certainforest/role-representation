All models can do the labeled task.
For non-labeled: 
1.5B sucks at even the two person task. 1.5B instruct is not any better.

7B seems better than 7B instruct on the two person task? Both suck at three person task.

A reasoning model can do both task. By reasoning about the discourse structure.

If you ask about a non-existing person:
1.5B instruct will cite the transcirpt without differentiating roles.
7B instruct will attribute other speaker's statement to this non-existing person.
Reasoning model will still try to parse who says what but believing that this person exist because user asked about them.