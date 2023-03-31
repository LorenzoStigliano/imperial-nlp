![[Screenshot 2023-02-25 at 10.51.26.png]]

- Pre-trained model are general good models trained on many iterations

## Pre-training encoder-decoder models

- Encoder-decoder: Input is processed using an encoder, then output is generated using a decoder.
- Initial input sequence can be fully attended over using the encoder, getting the maximum information.
- Then, a dedicated decoder is used for generating the output sequence.
- Example: Particularly popular for machine translation - encoder and decoder can have separate vocabularies and focus on different languages.![[Screenshot 2023-02-25 at 10.55.54.png]]

### Pre-training

- How to pre-train? Can’t really do Masked Language Modelling any more, there isn’t a direct correspondence between input tokens and output tokens.
- Ideas:
	- Prefix language modeling
		- Input: Thank you for inviting me 
		- Target: to your party last week .
	- Sentence permutation deshuffling - try to recover the true underlying sentence
	- BERT-style token masking - try to guess what word to use for the MASK token
	- We can combine methods together
		- We can corrupt the original sentence in various different ways, then optimise the model to reconstruct the original sentences. Good for generation tasks.
		- For example, the BART model:![[Screenshot 2023-02-25 at 11.00.13.png]]
- Replace corrupted spans SpanBERT-like objective, adapted for decoding
	- Take the original text we get sequences of text and remove them 
	- The model needs to recover the text and special markers![[Screenshot 2023-02-25 at 11.02.30.png]]
	- More challenging:
		- We remove many words in each set
## Instructional training

- T5 (Text-To-Text Transfer Transformer) is trained using the span corruption unsupervised objective, along with a number of different supervised tasks.![[Screenshot 2023-02-25 at 11.05.06.png]]
- Reformulated different sequence to sequence tasks and trained the T5 model again
- Take instructions from different tasks encoded in the sentence and the model was trained to complete all of the tasks
- Prefixing the input with a particular string for a given task isn’t very natural.
- Instead existing datasets can be converted into more conversational-sounding instructions using templates.![[Screenshot 2023-02-25 at 11.07.59.png]]
- FLAN-T5 (Fine-tuned LAnguage Net) is the same size as T5 but trained on much more data, more languages and 1.8K tasks phrased as instructions.![[Screenshot 2023-02-25 at 11.09.26.png]]

## Pre-training decoder models

- They don’t have an explicit encoder. 
- Language models.
- Everything happens together in the decoder. Using efficient attention for generating one word at a time.![[Screenshot 2023-02-25 at 11.15.58.png]]
- We can train on unlabeled text, optimizing, next word given all of the previous words
- ![[Screenshot 2023-02-25 at 11.16.32.png]]
- Great for tasks where the output has the same vocabulary as the pre-training data.
- For example: dialogue systems, summarization, simplification, etc.

### Learning methods

- Alternative ways of using pre-trained decoders:
	1. Fine-tuning: Supervised training for particular input-output pairs. Or we can put a new layer on top and fine-tune the model for a desired task.![[Screenshot 2023-02-25 at 11.18.35.png]]
	2. Zero-shot: Give the model a natural language description of the task, have it generate the answer as a continuation.![[Screenshot 2023-02-25 at 11.18.45.png]]
	3. One-shot: In addition to the description of the task, give one example of solving the task. No gradient updates are performed.![[Screenshot 2023-02-25 at 11.18.54.png]]
	4. Few-shot: In addition to the task description, give a few examples of the task as input. No gradient updates are performed.![[Screenshot 2023-02-25 at 11.20.59.png]]

### Fine-tuning decoder models

- Once pre-trained, we can fine-tune these models as classifiers, by putting a new output layer onto the last hidden layer.
- The new layer should be randomly initialised and then optimized during training.
- We can backpropagate gradients into the whole network.![[Screenshot 2023-02-25 at 11.24.06.png]]
- The original GPT performed generative pre-training of the decoder but then was fine-tuned as a discriminative classifier.
- Natural Language Inference: 
	- Label pairs of sentences as entailing/contradictory/neutral
	- Premise: The man is in the doorway 
	- Hypothesis: 
	- The person is near the door 
	- This input is formatted, as a sequence of tokens for the decoder: 
		- [START] The man is in the doorway [DELIM] The person is near the door [EXTRACT] 
	- The linear classifier is applied to the representation of the [EXTRACT] token.
	- We have it at the end since decoders process left to right
- GPT was a big step in the area of pre-trained decoders
	- Transformer decoder with 12 layers, 117M parameters.
	- 768-dimensional hidden states, 3072-dimensional feed-forward hidden layers.
	- Byte-pair encoding with 40,000 merges 
	- Trained on BooksCorpus: over 7000 unique books
	- Contains long spans of contiguous text, for learning long-distance dependencies
- Pre trained and then using the [EXTRACT] method improved the methods

## Zero-shot learning

- A key emergent ability in large language models is zero-shot learning
- The ability to do many tasks with no examples, and no gradient updates, by simply
	- Specifying the right sequence prediction problem
	- Comparing probabilities of sequences, calculate the probability of a sequence using the language model, where the sequence is generated by the model

## Few-shot learning

- Large language models are also able to learn from examples.
- Simply give it some examples of doing a task and have it continue the generation.
- For best results, this is combined with the task description.
- *Requires no gradient updates! The model stays frozen and only learns from the input that is given.*![[Screenshot 2023-02-25 at 11.38.56.png]]
- These are methods that are used to get better performance by changing the prompts given to the models
- These models pick up biases in the data 