- Goal: tag a sentence with different tags
- We have open class and closed class tags 
- Why do we need PoS tagging?
	- Name entity recognition entities are nouns
	- Pre-processing can can be based on PoS
	- For neural syntactic and semantic parsing

1. Naive approach
	- Assign each word its most frequent POS tag
	- Assign all unknown words the tag NOUN
	- 90% accuracy
2. Probabilistic POS tagging
	- Use frequencies but take context into account![[Screenshot 2023-02-27 at 16.10.57.png]]
	- Generative approach 
		- Assumptions![[Screenshot 2023-02-27 at 16.11.35.png]]![[Screenshot 2023-02-27 at 16.13.10.png]]
		- Greedy approach take the max probability at each step 
	- HMM tagger 
		- Issues with tagging left to right based on local max?
			- We are ignoring more promising paths overall by sticking to one decision at a step
		- ![[Screenshot 2023-02-27 at 16.20.19.png]]
		- Markov Chains: 
			- model probabilities of sequences of random variables (states)
		- Hidden Markov Chains: states are not given, but hidden 
			- Words are observed 
			- POS are hidden
		- Formal set up ![[Screenshot 2023-02-27 at 16.21.41.png]]
		- Assumptions![[Screenshot 2023-02-27 at 16.22.59.png]]
		- Formulation ![[Screenshot 2023-02-27 at 16.23.32.png]]
		- Decoding/inference: task of determining the hidden state sequence corresponding to the sequence of observations![[Screenshot 2023-02-27 at 16.24.53.png]]

## Viterbi algorithm 

- Used to solve the above maximisation problem 
- Dynamic programming
- First build a lattice/matrix
	- One column per observation and one row per state 
	- Each node $v_t(j)$ is the probability that the HMM is in state $j$ after seeing the first $t$ observations and passing through the most probable state sequence $q_1 , ..., q_{t−1}$![[Screenshot 2023-02-27 at 16.28.27.png]]![[Screenshot 2023-02-27 at 16.29.13.png]]
	- What sequence of tags is the best?
		- Start from end, trace backwards all the way to beginning
		- This gives us the chain of states that generates the observations with the highest probability
- Computational complexity 
	- Viterbi’s running time is $O(SN^2 )$, where $S$ is the length of the input and $N$ is the number of states in the model
	- Some tagsets are very large: 50 or so tags
		- Beam search as alternative decoding algorithm - only expand on top k most promising paths

### MEMM for PoS tagging 

- HMM is a generative model, powerful but limited in the features it can use
- Alternative: sequence version of logistic regression classifier *Discriminator*
	- maximum entropy classifier (MEMM), a discriminative model to directly estimate posterior![[Screenshot 2023-02-27 at 16.40.19.png]]
![[Screenshot 2023-02-27 at 16.41.29.png]]

## RNN for POS tagging

- RNN to assign a label from (small) tagset to each word in the sequence
- Inputs: word embedding per word 
- Outputs: tag probabilities from a softmax layer over tagset
- RNN: 1 input, 1 output, 1 hidden layer; U, V and W shared

## SOTA

- Using Universal Dependencies (UD) as tagset for various languages
- Ensemble of subword representations
- Big models not considerable results with HMM 


