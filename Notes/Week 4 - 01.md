## Language models

### 1. N-gram models

- Language modeling involves assigning probabilities to sequences of words.
	- Predicting the next word in a sequence of words
	- Predicting a masked word in a sentence

Why?
- Some tasks require generating language responses, rather than choosing a specific class: Word completion, Machine translation, ....
- To perform language modelling, we need to consider language as a sequence.

#### N-gram modelling

- We aim to compute $P(w|h)$ where , $w$ is the word and $h$ is the history $P(w_n|w_{1}^{n-1})$, all h words before w
- Problem: long grams there are not enough examples.
- Solution: N-gram models approximate history by just the few last words![[Screenshot 2023-01-30 at 16.27.17.png]]
- Unigram, probability of a word
- Assumption: N-gram models approximate history by just the few last words
- Estimating probabilities: “MLE as relative frequencies”
- Corpus size: 
	- The larger, the better the counts - larger n possible 
	- Trigrams are often enough

#### Evaluating language models
- Decompose the problem
	- Estimate joint probability of an entire word sequence by multiplying together a number of conditional probabilities![[Screenshot 2023-01-30 at 16.36.29.png]]![[Screenshot 2023-01-30 at 16.37.02.png]]![[Screenshot 2023-01-30 at 16.43.24.png]]
- We have an issue here with longer outputs:
		- The longer the output is, the lower its likelihood, so very long outputs will have a very small likelihood.
- Solution, **perplexity**:
		- It’s the inverse probability of a text, normalised by the # of words![[Screenshot 2023-01-30 at 16.44.18.png]]
		- Minimising perplexity -> maximising probability
		- t’s a measure of the surprise in a LM when seeing new text
		- Perplexity allows us to choose the best LM for a test data
		- LM1 vs LM2: best LM is the one with the lowest perplexity
		- Perplexity is specific to the test-set

#### Cross Entropy Loss
- We don’t know the true distribution… e.g. the likelihood of each possible next word
- We only know how many times things happen in the training data![[Screenshot 2023-01-30 at 16.47.59.png]]
![[Screenshot 2023-01-30 at 16.49.12.png]]
- If we are finding the perplexity of a single word, what is the best possible score? 1
- If our model uniformly picks words across a vocabulary of size |V|, what is the perplexity of a single word? Size of vocabulary $\frac{1}{1/|V|} = |V|$ $(n = 1)$
- Our LM predicts digits between 0 and 9 as our words with even probability: 
	- what is the perplexity if our test-set contains 5 words? $p = \frac{1}{10}$

#### Extrinsic vs Intrinsic evaluation
- If the goal of your language model is to support with another task 
	- The best choice of language model is the one that improves downstream task performance the most (extrinsic evaluation) 
- Perplexity is less useful in this case (intrinsic evaluation)

#### Evaluating GPT-3

Language models, such as GPT3, can also be evaluated on their ability to perform a range of classification tasks: 
- Question / answer tasks with multiple choices 
- Finding the most likely end to a sentence / short story 
- Performing tasks such as Natural Language Inference

#### Language models: sparsity

Techniques to mitigate sparsity:
- Add-1 Smoothing 
	-  Given words with sparse statistics, steal probability mass from more frequently words![[Screenshot 2023-01-30 at 17.24.25.png]]![[Screenshot 2023-01-30 at 17.24.07.png]]
	- Easy to implement  
	- But takes too much probability mass from more likely occurrences 
	- Assigns too much probability to unseen events 
	- Could try +k smoothing with a smaller value of k
- Back-off 
	- If we do not have any occurrences of a ‘his royal highness’: 
	- We could back-off and see how many occurrences there are of ‘royal highness’
	- 0.4 hyperparameter![[Screenshot 2023-01-30 at 17.25.31.png]]
- Interpolation
	- ![[Screenshot 2023-01-30 at 17.25.45.png]]
#### Evaluation
- Train LM on 38 million words of WSJ
- Test on 1.5 million held-out words also from WSJ
- Choose LM with smallest perplexity 
- N-gram LM: good approximation of language likelihood

