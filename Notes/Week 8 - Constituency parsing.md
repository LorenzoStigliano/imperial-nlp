
- Given a sequence of words (usually a sentence), generate its syntactic structure
- Applications
	- Grammar checking 
	- Semantic analysis
		- Question answering
		- Named entity recognition
	- Machine translation
- Challenges -> one sentence can have many different syntactic structures 

## Classical Parsing 

- Given a grammar and a lexicon and a new sentence
- Goal: Generate the structure for the sentence![[Screenshot 2023-02-27 at 17.08.52.png]]
- To do this automatically 
	- Use proof systems to prove parse trees from words
	- Search problem: all possible parse trees for string

### The CKY algorithm
- Tests for possibilities to split the current sequence into two smaller sequences
- Grammar needs to be in Chomsky Normal Form (CNF) ![[Screenshot 2023-02-27 at 17.10.04.png]]
- Binarisation makes CKY very efficient
	- $O(n^3 |G|)$: $n$ is the length of the parsed string; $|G|$ is the size of the CNF grammar $G$ 
	- Otherwise it would be exponential
- Dynamic programming algorithm to efficiently generate all possible parse trees bottom-up
- ALGORITHM - we need to always combine words into two substrings![[Screenshot 2023-02-27 at 17.12.23.png]]

### CKY for parsing
1) Add backpointers: augment entries in matrix s.t. each non-terminal is paired with pointers to the matrix entries from which it was derived 
2) Allow multiple copies of the same non-terminal to be entered into the matrix to track multiple paths
3) Choose S from cell [0, n] and recursively retrieve its component constituents from the matrix

- This does not scale: too many possible parse trees for comprehensive grammar

## Constituency parsing

- Based on the idea of phrase structure - Words are grouped into constituents
- A constituent is a sequence of words that behaves as a unit, generally a phrase

## Statistical parsing

- Solution:
	- Find the most likely parse(s) via statistical parsing 
	- Comprehensive grammars admit many parses for a sentence, but we can efficiently find the most likely parse
- “Learn” probabilistic grammars from labelled data: treebanks
- Treebanks are expensive to build, but:
	- Frequencies and distributional information are important 
	- Can be reused to build different parsing approaches 
	- Provide a way to evaluate parsers
- What does it mean to “learn” a grammar?
	- Take annotated trees (e.g. Penn Treebank) and List all rules used
	- Assign probabilities to all rules by counting![[Screenshot 2023-02-27 at 17.35.48.png]]
![[Screenshot 2023-02-27 at 17.36.09.png]]
- Intuition all the rules are used independently as a result we get the most probable tree ![[Screenshot 2023-02-27 at 17.39.47.png]]![[Screenshot 2023-02-27 at 17.39.57.png]]

### The CKY algorithm for PCFG
![[Screenshot 2023-02-27 at 17.42.06.png]]
![[Screenshot 2023-02-27 at 17.42.29.png]]
![[Screenshot 2023-02-27 at 17.43.55.png]]
![[Screenshot 2023-02-27 at 17.44.44.png]]
![[Screenshot 2023-02-27 at 17.45.07.png]]

### Issues with PCFG

- **Poor independence assumption**: CFG rules impose an independence assumption on probabilities that leads to poor modelling of structural dependencies across the parse tree. Word is only dependent on its POS tag! 
	- Probability estimates of rules computed independently of surrounding context
	- We can capture this with -> Split non-terminals![[Screenshot 2023-03-02 at 11.21.51.png]]![[Screenshot 2023-03-02 at 11.22.31.png]]
- **Lack of lexical conditioning**: CFG rules don’t model syntactic facts about specific words, leading to problems with subcategorization ambiguities, preposition attachment, and coordinate structure ambiguities
	- Probabilistic Lexicalised CFGs
		- Add annotations specifying the head of each rule 
		- Each rule in the grammar identifies one of its children to be the head of the rule![[Screenshot 2023-03-02 at 11.25.06.png]]![[Screenshot 2023-03-02 at 11.27.23.png]]
		- Head is core linguistic concept, the central sub-constituent of each rule
		- Treebanks not annotated for that, but can use rules to identify head
## Evaluating parses

- Parseval metrics: evaluate structure
	- How much of constituents in the hypothesis parse tree look like the constituents in a gold-reference parse tree
	- A constituent in hyp parse is labelled “correct” if there is a constituent in the ref parse with the same yield and LHS symbol
		- Only rules from non-terminal to non-terminal
	- Metrics are more fine-grained than full tree metrics, more robust to localised differences in hyp and ref parse trees![[Screenshot 2023-03-02 at 11.14.39.png]]

