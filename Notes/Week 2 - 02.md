## Word representations

### Word as vectors

- Represent words as vector so that similar meanings have similar vector.

#### 1-hot vectors

- Each element represents a different word. 
- Have “1” in the position that represents the current word. 
- Otherwise the vector is full of zeros. 
- Also known as “1-hot” or “1-of-V” representation.

Problems with 1-hot vectors
- We can’t fit many and they tell us very little. 
- Need a separate dimension for every word we want to represent (170K+). 
- All the vectors are orthogonal and equally distant.

#### Mapping of words to concepts

- Map words to broader concepts using wordnet: http://wordnetweb.princeton.edu/perl/webwn or https://www.nltk.org/howto/wordnet.html
- Reduces similar words together
- Reduces vector size

Problems with words to concepts
- Relies on completeness of a manually curated database
- Misses out on rare new meanings
- Vectors **still** orthogonal and equally distant from each other
- Disambiguation is complicated: mouse: animal or device?

#### Distributed vectors

- Each element represents a property and these are shared between the words.![[Screenshot 2023-01-19 at 11.13.20.png]]
- Then we can use cosine to calculate the similarity between two words
- 1 in the same direction, 0 unrelated orhtogonal, -1 oposite direction of vectors
![[Screenshot 2023-01-19 at 11.14.13.png]]

- We can infer some information based only on the vector of the word 
- We don’t even need to know the labels on the vector elements

### Distributed hypothesis
- Words which are similar in meaning occur in similar contexts.

#### Count-based vectors
- count how often a word occurs together with specific other words (within a context window of a particular size).

#### TF-IDF
- Problem: Common words (“a”, “the”, “it”, “this”, …) dominate the counts
- Can weight the vectors using $TF-IDF = TF * IDF$
- Term Frequency (TF): Upweights words $w$ that are more Important to $d$
- Inverse Document Frequency (IDF): Downweights words that appear everywhere

![[Screenshot 2023-01-19 at 11.19.47.png]]

### Word Embeddings
- The count-based vectors are still very large (possibly 170K+ elements) 
- They are also very sparse - mostly full of zeros

- Soution: allocate a number of parameters for each word and allow a neural network to automatically learn what the useful values should be
- Referred to as “word embeddings”, as we are embedding the words into a real-valued low-dimensional space
- The resulting vectors are usually short (~300-1K) and dense (non-zero)

#### Continous Bag-of-words (CBOW)
![[Screenshot 2023-01-19 at 11.31.07.png]]

#### Skip-gram
- Predict the context words based on the target word $w_t$ it uses two embeddings for which we can define the size and shape.
![[Screenshot 2023-01-19 at 11.31.43.png]]
- Give 1-hot vector of the target word as input $x$
- Map that to the embedding of that target word, using weight matrix W: $h=xW$
- Map the embedding of the target word to the possible context words, using weight matrix W’: $y=hW'$
- y has length of whole vocabulary. Apply softmax to make y into a probability distribution over the whole vocabulary, $y' = softmax(y)$
![[Screenshot 2023-01-19 at 11.34.53.png]]
- The whole model is just two matrices of embeddings.
- Directly optimizing for the embedding of the target word from W to be similar to the embedding of the context word from W’.
![[Screenshot 2023-01-19 at 11.39.43.png]]
![[Screenshot 2023-01-19 at 11.40.16.png]]

- **Downside** In order to compute a single forward pass of the model, we have to sum across the entire corpus vocabulary (in the softmax)
- This gets inefficient with large vocabularies and large embeddings

To optimize:
- Instead of vector multiplication we can use word embedding as a DB
- Instead of multi-class classification, train the model using a binary classification objective (logistic regression) to discriminate real context words ($W_{t+1}$ ) from $k$ other (noise) words
- **Negative Sampling**
![[Screenshot 2023-01-19 at 11.44.53.png]]![[Screenshot 2023-01-19 at 11.45.49.png]]
How to select which words to use as negative examples?
- Randomly or by frequency (selecting more frequent words more frequently) 
- 5-20 negative words works well for smaller datasets 
- 2-5 negative words works well for large datasets

#### Drawbacks 
- Require large amounts of data to train 
- Low quality for rare words 
- No coverage for unseen words 
- Antonyms tend to have similar distributions: e.g. “good” and “bad” 
- Does not consider morphological similarity: “car” and “cars” are independent 
- Does not differentiate between different meanings of a word