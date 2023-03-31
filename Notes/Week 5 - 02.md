
## Attention architecture
![[Screenshot 2023-02-09 at 11.24.46.png]]

## Encoder hidden states
![[Screenshot 2023-02-09 at 11.05.33.png]]
- We project the last hidden state to obtain a D dimensional 
- Now we’re looking at the decoding process. The decoder RNN has inputs: INIT vector, and the SOS token
- Notice 2D since we are using bidirectional RNN, where D is the dimension of the hidden state![[Screenshot 2023-02-09 at 11.06.30.png]]
- Before we generate the next predicted word, we run the attention mechanism 
- The attention mechanism takes ALL the encoder hidden states as input, and the decoder hidden state being inputted to the RNN to decode the current word![[Screenshot 2023-02-09 at 11.07.10.png]]
- Attention is then performed over these inputs, and the module outputs a context vector specific to this decoding time step
- The RNN now has 3 inputs: The hidden state (INIT in this case), a context vector specific to the current decoding step (c_0), and the input token (SOS)
- Using these 3 inputs, it gives us 2 outputs: 1) Our decoder hidden state vector (s_0), and also a prediction of the first word (We)
- We then start decoding the next word. If we’re using teacher forcing, we feed in the correct ground truth word as input (instead of the prediction)![[Screenshot 2023-02-09 at 11.08.04.png]]![[Screenshot 2023-02-09 at 11.08.29.png]]
- Again, we run the attention process. 
- We take in ALL encoder hidden states as input, and the previously outputted decoder hidden state
- The projection is used in the INIT state

## Attention process
![[Screenshot 2023-02-09 at 11.08.57.png]]
## Energy scores come from the alignment function
![[Screenshot 2023-02-09 at 11.10.08.png]]
- We’re going to obtain an energy score for EACH source word
- Energy scores are obtained using an alignment function, a  
- a() takes 2 inputs: The previous decoder hidden state (s_t-1) and the encoded hidden state for ONE word (h_i) 
- Just to be clear - this function is run for every source word we have
## Alignment function![[Screenshot 2023-02-09 at 11.11.12.png]]
- The previous decoder hidden state s_t-1 is a D dimensional vector.
- The encoder hidden state for ONE word (the i’th word), is a 2D dimensional vector
- This scalar is e_i, the unnormalized energy score![[Screenshot 2023-02-09 at 11.13.04.png]]
- We run the above for all of the words in the sentence ![[Screenshot 2023-02-09 at 11.14.47.png]]![[Screenshot 2023-02-09 at 11.14.55.png]]
- Currently our energy scores are unnormalized 
- We normalize them by sending all the energy scores to a softmax function.
- This gives us alpha values between 0 and 1

## Calculating $c_t$ from our alphas
![[Screenshot 2023-02-09 at 11.15.58.png]]
![[Screenshot 2023-02-09 at 11.16.26.png]]
![[Screenshot 2023-02-09 at 11.16.38.png]]
- Continuing from where we were in our diagram, we then output the context vector for the t=1 word (c1) 
- This forms our 3rd input to the current RNN step (alongside s_0 and an input word)
- We concatenate c_1 s_0 and teacher I
- **And the process repeats until we hit an EOS token**

## Evaluation metrics of MT/ Natural Language Generation systems

- Human evaluation
	- Best but expensive
- Automatic evaluation
	- Fast/proxy measures for evaluation
	- Often based on count stastics of n-grams
	- BertScore is a non n-gram and model based evaluation metric

### BLEU
- BLEU reports a modified precision metric for each level of n-gram
- Credit is assigned as the maximum amount of times each unique n-gram appears over all the references 
- MP is modified precision = total unique overlap / total MT ngrams
- Total unique overlap - how many times does the ngram appear in the reference sentences and see if we see more overlap over the references
![[Screenshot 2023-02-09 at 11.31.03.png]]
1. First we work out the number of unique ngrams in our Hyp. MP-1 means we’re looking at unigrams (MP-2 would be bigrams, etc) 
2. Then we look at the ngram overlap between the unique ngrams in our Hyp and all ngrams in our Refs. 
3. Work out the number of unique overlaps between the unique ngrams and all the Refs 
4. Then we obtain our modified-precision scores: the total unique overlap divided by the number of ngrams in our Hyp![[Screenshot 2023-02-09 at 11.40.29.png]]![[Screenshot 2023-02-09 at 11.41.16.png]]

- There are a couple of interpretations about why we use a BP. They’re mostly about encouraging the Hyps to be of a similar length to a reference (see BP equation). Feel free to research more about it in your own time. An intuitive reason is for its existence is to account for the lack of recall term in the metric.
- Practically, there are some differing definitions and implementations of BLEU. When you want to report this score, it is good practise to use a standardized library (e.g. SacreBLEU)

### Chr-F and TER 
 - Chr-F is an F beta -score based metric over character n-grams. This metric balances character precision and character recall
 - TER is performed at the word level, and the “edits” can be a: Shift, Insertion, Substitution and Deletion. TER balances these edits to build a metric![[Screenshot 2023-02-09 at 11.42.47.png]]

### ROUGE 
- a shortcoming of BLEU is that it focuses a lot on the precision between Hyp and Ref, but not the recall
- ROUGE balances both precision and recall via the F-Score
- ROUGE-n measures the F-score of n-gram split references and system outputs![[Screenshot 2023-02-09 at 11.43.48.png]]
- ROUGE-L is the F-score of the LCS between the references and system outputs. The subsequence does not have to be consecutive
- Other variants of ROUGE such as ROUGE-S, ROUGE-W which incorporate skip grams and/or weighting
### METEOR
- METEOR is more modern and a better metric than BLEU, though for some reason the NMT community still adopts BLEU a lot more frequently. You will find METEOR in other generation tasks such as summarization and captioning.![[Screenshot 2023-02-09 at 11.45.16.png]]![[Screenshot 2023-02-09 at 11.45.59.png]]

### BertScore
![[Screenshot 2023-02-09 at 11.48.08.png]]

## Inference 

![[Screenshot 2023-02-09 at 11.49.19.png]]
![[Screenshot 2023-02-09 at 11.49.38.png]]
![[Screenshot 2023-02-09 at 11.50.02.png]]
- In practise, k is normally between 5-10. For the example we’re about to work through, k=2
- Note that we will end up with k hypothesis at the end of decoding. Decoding finishes when we hit an EOS token![[Screenshot 2023-02-09 at 11.53.02.png]]

## Tricks

### Data augmentation
![[Screenshot 2023-02-09 at 11.54.13.png]]
![[Screenshot 2023-02-09 at 11.55.24.png]]