- Understand how the words affect each other in contrast on how they group together
- Connect words in sentence to indicate dependencies between them
- Build around notion of having heads and dependents
- Arrows can be annotated by different types of dependencies
	- Head (governor), also called argument: origin 
	- Dependent, also called modifier: destiny![[Screenshot 2023-03-02 at 11.34.59.png]]
- There are versions without typed dependencies, just arcs
	- Untyped variant is simpler to build but less informative![[Screenshot 2023-03-02 at 11.35.46.png]]

### Main differences to constituency parsing

- No nodes corresponding to phrasal constituents or lexical categories 
- The internal structure consists only of directed relations between pairs of lexical items 
- These relationships allow directly encoding important information

### Main advantages in dependency parsing

- Ability to deal with languages that are morphologically rich and have a relatively free word order
- Would have to represent two rules for each possible place of the adverb for constituency
- Dependency approach: only one link; abstracts away from word 
- Head-dependent relations provide approximation to semantic relationship between predicates and arguments
- Can be directly used to solve problems such as co-reference resolution, question answering, etc.

## Dependency formalisms

- A dependency structure is a directed graph G = (V, A) 
	- V = set of vertices (words, punctuation, ROOT) 
	- A = set of ordered pairs of vertices (i.e. arcs)
- ![[Screenshot 2023-03-02 at 11.40.27.png]]
- This ensures: 
	- Dependency structure becomes a tree 
	- Each word has a single head 
	- The dependency tree is connected 
	- There is a single ROOT from which a unique directed path follows to each word in sentence

## Sources of information

- Distance between head and dependent
- Intervening material - Dependencies don’t cross over verbs or punctuation
- Valency of verbs - For a typical word, what kind of dependency it generally takes?

## Approaches 

- Dynamic programming
- Shift-reduce (transition-based)
	- Predict from left-to-right
	- Fast (linear), but slightly less accurate
- Spanning tree (graph-based, constraint satisfaction)
	- Calculate full tree at once 
	- Slightly more accurate, slower
- Deterministic parsing, shift-reduce (MALT parser)
	- Greedy choice of attachment for each word in order, guided by ML classifier
	- Works very well in practice

## MALT parser
![[Screenshot 2023-03-02 at 11.43.19.png]]![[Screenshot 2023-03-02 at 11.45.21.png]]
- How do we make the shift/reduce left/right decisions?
	- ML classifier
	- Each action is predicted by a discriminative classifier over each move
	- 3 classes for untyped dependencies: shift, left or right
	- 2categories + 1 for typed dependencies
	- Features: top of stack word, its POS; first in buffer word, its POS; etc![[Screenshot 2023-03-02 at 11.47.10.png]]

### Neural parser
- Follow-up work
- Replace binary features by embeddings (words & POS)
- Concatenate these embeddings
- Train a FNN as classifier with cross-entropy loss![[Screenshot 2023-03-02 at 11.48.38.png]]

## Evaluation 
![[Screenshot 2023-03-02 at 11.49.50.png]]
![[Screenshot 2023-03-02 at 11.50.06.png]]


## Neural parsing
![[Screenshot 2023-03-02 at 11.51.36.png]]
- Parsing as translation
	- Tried RNN and get entire sentence separated by brackets
	- Linearise grammar from treebank: convert tree to bracketed representation, all in one line
	- Extract sentences from bracketed representation
	- Pair sentences and their linearised trees
	- No need to compute/represent probabilities - learning
	![[Screenshot 2023-03-02 at 11.52.56.png]]
- New methods: Graph-based methods

## Takeaways

- Parsing is an important step for many applications
- Statistical models such as PCFGs allow for resolution of ambiguities
- Current statistical/neural parsers are quite accurate (~95% dependency; 97% constituency), Human-expert agreement: ~98%