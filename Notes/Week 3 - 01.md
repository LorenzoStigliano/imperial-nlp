### Classification

#### NLP Classification tasks

- Goal we want to find: $\hat{y} = argmax_yP(y|x)$
- Examples: binary classification, multi-class classification speech classification, sentiment analysis, fact verification, spam detection, error detection

#### Natural Language Inference 
- Labels:
	- Entailment: if the hypothesis is implied by the premise
	- Contradiction: if the hypothesis contradicts the premise
	- Neutral: otherwise

#### Naive Bayes
$$ \text{Bayes Rule: }P(y|x) = \frac{P(x|y)P(y)}{P(x)}$$
- Naive Bayes key assumption: Independence assumption.
	- $P(x_1,x_2,...x_n|y) = p(x_1|y)\times...\times p(x_n|y)$
- Input representation: Bag of Words representation, count number of word for the set of documents given.
- $y$ is the class, $x$ is the document
- From the training examples we can evaluate:
	- $p(y)$ probability of class
	- $p(x)$ is the probability that the word occurs in corpus = number of documents it appears in over number of documents
	- $p(x_i|y)$ we can Add-one smoothing, to ensure we have non zero probabilities![[Screenshot 2023-01-23 at 16.43.19.png]]

Improvements
- Include logical negation until next punctuation
- Count features once per document, is the word present or not 

Problems
- Conditional Independence assumption
- Context not taken into account
- New words not seen in training cannot be used

#### Logistic Regression

- Discriminative algorithm to directly learn what features from the input are most useful to discriminate between the different classes
- Loss function: cross entropy loss![[Screenshot 2023-01-23 at 16.44.28.png]]
- Weights are learnt per class
- Probability distribution over classes, use softmax for final output![[Screenshot 2023-01-23 at 16.45.48.png]]

##### When might you use them?
- They can help us to understand which features are influential or correlate with each class 
- This can help us better understand our dataset and which features are important
- We can compare to more powerful models to understand the nature of the task

#### Neural Networks (NN)
![[Screenshot 2023-01-23 at 16.57.52.png]]
![[Screenshot 2023-01-23 at 16.59.35.png]]![[Screenshot 2023-01-23 at 16.59.48.png]]
##### How to get a document representation of sentence of fixed dimensionality?

- We could find an average vector for all the words in the document.
- Problem, does not capture position of words in a sentence

DO NOT DO THIS:
- Model architecture fixed to sentence length size
- Model weights learnt for specific word positions![[Screenshot 2023-01-23 at 17.02.08.png]]
##### Why neural networks
- Automatically learned features 
- Flexibility to fit highly complex relationships in data
- But: they require more data to learn more complex patterns

#### Recurrent Neural Networks (RNNs)
- Natural language data - sequences
- Value of a unit depends on own previous outputs
- Usually the last hidden state is the input to the output layer![[Screenshot 2023-01-23 at 17.09.37.png]]
- RNN (f) computes its next state $h_{t+1}$ based on: Hidden state vector and input vector at time $t$![[Screenshot 2023-01-23 at 17.10.50.png]]
- Its hidden state is carried along (memory), we only need to learn one W and U matrix. ![[Screenshot 2023-01-23 at 17.11.31.png]] 
- Unrolling an RNN yields a deep feed-forward network![[Screenshot 2023-01-23 at 17.12.20.png]]![[Screenshot 2023-01-23 at 17.16.05.png]]
- We finish on a linear layer

##### Problems
- Vanishing gradient problem: The model is less able to learn from earlier inputs.
	- Tanh derivatives are between 0 and 1
	- Sigmoid derivatives are between 0 and 0.25
- Gradient for earlier layers involves repeated multiplication of the same matrix W
	- Depending on the dominant eigenvalue this can cause gradients to either ‘vanish’ or ‘explode’

#### Convolutional Neural Networks (CNNs)
- CNNs are composed of a series of convolution layers, pooling layers and fully connected layers [[Deep Learning/Notes/Week 2]] [[N01_Convolutions.pdf]]
- CNN learns values of its filters based on task
- Filter: sliding window over full rows (words) in one direction
	- Filter width = embedding dimension
	- Filter height = normally 2 to 5 (bigrams to 5-grams)![[Screenshot 2023-01-23 at 17.29.03.png]]![[Screenshot 2023-01-23 at 17.29.24.png]]![[Screenshot 2023-01-23 at 17.29.47.png]]![[Screenshot 2023-01-23 at 17.30.02.png]]![[Screenshot 2023-02-09 at 13.21.56.png]]
- CNNs can perform well if the task involves key phrase recognition 
- RNNs perform better when you need to understand longer range dependencies
- We define the number of channels as the number of features we want, the kernel size is usually height = n-gram and width = word embedding dimension
