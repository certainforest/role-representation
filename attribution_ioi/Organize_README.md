Assumptions of how models does the speaker tracking entity-binding task:

Each Example will produce a set of four experiment results:
Q_name_bind
Q_name_control
Q_country_bind
Q_country_control

Q_name means querying entity "Where does Alice live".
Q_country means querying attribute "Who lives in France".
kind: 
- Binding is when we change the lines corresponding to the entity or attribute being queried. (eg. when querying entity, change the "I am Alice/Bob" line) This is when we can observe a spike at the query token when doing activation patching. 
- Control is we when we change the lines that does not correspond to the entity or attribute being queried. (eg. when querying entity, change the "I live in France/Thailand" line) This is when we CANNOT observe any spike at the query token when doing activation patching. 