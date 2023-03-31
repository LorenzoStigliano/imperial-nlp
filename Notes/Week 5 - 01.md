## Statistical Machine Translation

- A pipeline of several sub-models:
	- Alignment model
	- Translation model
	- Language model

- Noisy channel model - uses parallel corpora as a training set
- Parallel corpus means we have aligned sentences: i.e. an english sentence and its corresponding french sentence
![[Screenshot 2023-02-06 at 16.43.21.png]]
- We want to model $p(t|s)$ $t$ is the input sentence given output sentence $s$
- Translation model contains phrases alongside their translation: 
	- Alignment model is responsible for extracting the phrase pairs 
	- Translation model is essentially a lookup table. We won’t dive into it, but you can imagine that statistics over large pairs of parallel texts can help identify parallel phrases 
	- Language model contains the probability of target language phrases
- We identify multiple candidate phrases 
- Language model $p(t)$ then says: for each of the candidate targets, how likely is that phrase to occur in the target language.
![[Screenshot 2023-02-06 at 16.44.12.png]]
Downsides: 
- Sentence alignment: 
	- In parallel corpora single sentences in one language can be translated into several sentences in the other and vice versa. Long sentences may be broken up, short sentences may be merged. There are even some languages that use writing systems without clear indication of a sentence end (for example, Thai). 
- Word alignment:
	- Is about finding out which words align in a source-target sentence pair - One of the problems presented is function words that have no clear equivalent in the target language. For example, when translating from English to German the sentence "John does not live here," the word "does" doesn't have a clear alignment in the translated sentence "John wohnt hier nicht."
- Statistical anomalies: 
	- Real-world training sets may override translations of, say, proper nouns. An example would be that "I took the train to Berlin" gets mis-translated as "I took the train to Paris" due to an abundance of "train to Paris" in the training set. 
- Idioms: 
	- Only in specific contexts do we want idioms to be translated. For example, using in some bilingual corpus (in the domain of Parliment), "hear" may almost invariably be translated to "Bravo!" since in Parliament "Hear, Hear!" becomes "Bravo!”.

## Neural Machine Translation

- Encoder decoder architectures
- Used to solve sequence to sequence tasks 
- We have an encoder function and then we use a decoder to unravel the input then we use 
- The encoder represents the source as a latent encoding
- The decoder generates a target from the latent encoding 

Types
1. RNNs - BiDirectional RNN
	- Using a BiRNN is appropriate for encoding the source as we’re not trying to predict the source. We have access to the whole thing at inference time. 
	- Using a BiRNN, and reading information from both sides, allows us to gain a more information dense representation of our sequence.
	- Call c the encoding of our source 
	- Feed c as the initial state to a decoder RNN
	- ![[Screenshot 2023-02-06 at 16.25.21.png]]
	- Think of the decoder as a typical language model (that you looked at last week). The only difference is that we initialise it with our c vector instead of a zerovector
	- Think that we input into the decoder the last element of each RNN.
1. Teacher Forcing 
	- Decoder is auto-regressive only during inference
	- During training, we use teacher forcing with a *ratio*. I.e. we well teacher force ratio% of time. The ratio can be 100% (i.e. full teacher forcing), 50%, or you can even anneal it during training.
	- We feed the ground truth at training time 
	- With standard autoregressive modelling we will most likely accumulate errors so, thats why we feed the truth 
	- Teacher forcing creates a phenomenon called Exposure Bias.
	- Exposure Bias is when the data distributions the model is conditioned on vary between training and inference
		- In this case, if we set teacher forcing to 100%, then the model is never conditioned on its own predictions 
		- So during inference, when the model uses the previously generated words as context, it is different than what it has seen in training
	- ![[Screenshot 2023-02-06 at 16.30.04.png]]
	- y_ < t means all the tokens up to the t’th timestep.

Why is this naive?
- All sequences are being represented by a d-dimensional vector. For longer sequence lengths, the ability to retain all the source information in this vector diminishes.
- From the decoder side of things, as more words get decoded and the hidden state gets updated to incorporate the previously generated words, information from the context vector may start to diminish.. 
- This is related to: 
	- vanishing gradients and the lack of ability to ‘remember’ things from earlier parts of the sequence

## Attention

- Think of it as a dynamic weighted average
- In the immediate context, it allows us to dynamically look at individual tokens in the input and decide how much weighting a token should have with respect to the current timestep of decoding. 
- Additive/MLP attention ![[Screenshot 2023-02-06 at 17.19.31.png]]
	- $c_t$ is the context vector for the t’th decoding timestep
	- We then loop over all the hidden states $i$, and weight it by a scalar value alpha
	- So if alpha is 0, then the i’th hidden state is 0
	- If alpha is 1, then we retain the full information of that hidden state
	- We then sum together our weighted hidden states to obtain a contextualised representation for the t’th decoding step

How do we get alpha?![[Screenshot 2023-02-06 at 17.21.02.png]]
1. What we’re trying to do is decode the y_t word
2. And we have access to bidirectional encodings of each source word 
3. Think of the attention module as a black box for a second. We’ll look at how it works in the next slide 
4. So, before we decode y_t, we’re going to feed all our encoder hidden states AND the decoder hidden state (s_t-1) to our attention module 
5. The module will output a context vector, c_t. 
6. c_t is a dynamic and contextualised representation. It uses the decoder hidden state information to try and figure out which source words are most important when for decoding y_t 
7. We send c_t to the t’th RNN step, alongside the previously generated token y_t1. 
8. One final change that the methodology introduced (not strictly related to attention itself), is that the output projection layer now also takes in c_t and the word embedding for y_t-1 (alongside s_t) to predict the t’th word.![[Screenshot 2023-02-06 at 17.22.53.png]]![[Screenshot 2023-02-06 at 17.24.37.png]]
- alignment $a$ is a two layer NN
- $h$ is a matrix, $Uh$ is $i\times D$
![[Screenshot 2023-02-06 at 17.32.38.png]]
- We can visualise our attention values to see which source words were looked at most when decoding a target word 
- I.e. when we’re decoding the word person, in German, we’re looking strongly at “person” in English, and also “the”.