## Byte Pair Encoding

### Subword units

- Neural models for NLP take sequences of tokens as input and map them to distributed embeddings. 
- When only dealing with complete words, there are some issues.![[Screenshot 2023-02-20 at 16.03.53.png]]
- We can break long and complex words down into smaller parts, so that each part occurs frequently enough in the training data to learn a good embedding for it.![[Screenshot 2023-02-20 at 16.06.30.png]]

### Byte Pair Encoding

- Instead of manually specifying rules for lemmatisation or stemming
- learn from data which character sequences occur together frequently
- Originally a compression technique![[Screenshot 2023-02-20 at 16.10.35.png]]
#### Training algorithm ![[Screenshot 2023-02-20 at 16.11.55.png]]
- Number of mergers is a hyperparameter
- The end-of-word symbol "\_"
	- Not actually an underscore (there are underscores in normal text). Could be any symbol or character that is not otherwise used in the normal vocabulary.
- *Why do we have it?*
	- It distinguishes the suffixes. 
		- “st” in “star” vs “st_” in “widest_”
	- It tells us where to insert spaces when putting words back together. 
		- “th” “is” “is” “a” “sen” “te” “nce” → “thisisasentence” 
		- “th” “is_” “is_” “a_” “sen” “te” “nce_” → “this is a sentence

#### Byte Pair Encoding: Inference Algorithm

- Once the model is trained, we receive a new word which we didn’t see during training![[Screenshot 2023-02-20 at 16.18.09.png]]

### Overview

- The individual characters are always in the vocabulary, so we can map even “hyzxixmsg” to “h”, “y”, “z”, “x”, “i”, “x”, “m”, “s”, “g”
- However, when dealing with Unicode characters, we may encounter characters that we haven’t seen during training.
- A variant of BPE works on sequences of bytes instead of characters. The base vocabulary size is then only 256 and we don’t have to deal with unseen characters

## Wordpieces

- Very similar algorithm to BPE - using corpus statistics to decide how to split words into subwords.
- Main differences:![[Screenshot 2023-02-20 at 16.23.27.png]]

## Contextual word representations
- We can learn a vector for each word, such that similar words have similar vectors.
- The meaning of a word can depend on its context
	- I deposited some money in the bank 
	- I was camping on the east bank of the river
- Having a single vector for every word doesn’t cut it. 
- Need to take the context into account when constructing word representations.

Idea: We can train a recurrent neural network to predict the next word in the sequence, based on the previous words in the context. Internally it will learn to encode any given context into a vector.![[Screenshot 2023-02-20 at 16.28.00.png]]

### ELMo: Embeddings from Language Models

- Take a large corpus of plain text and train two language models:![[Screenshot 2023-02-20 at 16.28.42.png]]

- When we need a vector for a word, combine the representations from both directions.![[Screenshot 2023-02-20 at 16.29.07.png]]
- 3 different representation of the word ![[Screenshot 2023-02-20 at 16.29.46.png]]
- ELMo could be integrated into almost all neural NLP tasks with a simple concatenation to the embedding layer.

## Pre-training encoders

### BERT

- Bidirectional Encoder Representations from Transformers (Devlin et al., 2019). 
- Takes in a sequence of tokens, gives as output a vector for each token. Builds on previous work (ELMo and others), combines some good ideas and scales it up further![[Screenshot 2023-02-20 at 16.37.08.png]]
- Transformers BERT architecture ^
- Input embeddings:
	- Transformers have no concept of sequence order. 
	- Positional embeddings allow it to capture the order of the input words.![[Screenshot 2023-02-20 at 16.39.50.png]]
- Token embeddings -> trainable embedding
- Segment embeddings -> If multiple sentences it tells the model to which sentence it belongs to 
- Positional embeddings -> Gives a sense of ordering of the input to the transformer 

### Masked Language Modeling

- We want to train this model on huge amounts of plain text, so it learns general-purpose language understanding.
- Use masked language modeling as the training task - it requires no labeled data! 
- Hide k% of the input words behind a mask, train the model to predict them![[Screenshot 2023-02-20 at 16.43.04.png]]
![[Screenshot 2023-02-20 at 16.48.12.png]]
- If we masked all the chosen words, the model wouldn’t necessarily learn to construct good representations for non-masked words.

## Putting pre-trained models to work

### How do we use these models?

- BERT-like models give us a representation vector for every input token. Just have to chop off the masked LM head, it is no longer needed.

Option 1: 
- Freeze BERT, use it to calculate informative representation vectors. 
- Train another ML model that uses these vectors as input. 

Option 2 (more common these days): 
- Put a minimal neural architecture on top of BERT (e.g. a single output layer) 
- Train the whole thing end-to-end (called fine-tuning).


### Sentence classification

- We can add a special token in the input that represents the whole sentence
![[Screenshot 2023-02-20 at 17.04.52.png]]

### Token labeling

- Putting a classification layer on top of token vectors![[Screenshot 2023-02-20 at 17.05.24.png]]

### Sentence pair classification

- Giving multiple sentences as input![[Screenshot 2023-02-20 at 17.05.50.png]]

### Question answering

- Labeling the tokens in the candidate answer span![[Screenshot 2023-02-20 at 17.06.07.png]]


## A whole family of pre-trained encoders
![[Screenshot 2023-02-20 at 17.14.58.png]]
- SpanBERT
	- Instead of random tokens, mask contiguous tokens and predict what is behind the masked span. 
	- Makes the task harder and the resulting model performs better.
- DistilBERT and ALBERT
	- For some applications, we need to create smaller and faster models. For example, using distillation: 
		- training a small model to behave similarly to the bigger version, while keeping most of the benefit.
- Big Bird and LongFormer
	- Regular self-attention has $O(N^2)$ complexity, so doesn’t scale very well. Models with sparse attention mechanisms have been proposed to extend the input length.
- ClinicalBert, MedBert, PubMedBert, BEHRT
	- There are BERT-like models trained on data from specific domains or designed for specific applications. For example, models for the medical domain.
- ERNIE
	- We have existing knowledge graphs and could take advantage of the information that they contain. 
	- Given that entities are detected in text, ERNIE adds special entity embeddings into the transformer for additional information. 
	- A separate multi-headed attention operates over the entity embeddings and then the information is combined with the main model in a fusion layer.

## Multimodal models
- Combining visual and textual information into the same transformer. LXMERT, VisualBERT, ImageBERT, VilBERT![[Screenshot 2023-02-20 at 17.21.33.png]]
- ImageBERT encodes detected objects as additional tokens. Uses 4 different training objectives: 
	- Masked language modeling 
	- Masked object classification
	- Predicting the features of a masked object 
	- Classifying whether image and text are related

### Multimodal models: ViLBERT
- ViLBERT uses two parallel BERT-like encoders. 
- These interact using a co-attention module. Key & value matrices are received from the other modality. 
- Image features are extracted using pre-trained Faster R-CNN (object detection model)

## Parameter-efficient fine-tuning

- Models fine-tuned for one task are usually better at that particular task, compared to models trained to do many different tasks. 
- We don’t want to have thousands of different copies of huge models, each one trained for a slightly different task. 
- Instead: let’s keep most of the parameters the same (frozen) and fine-tune only some of them to be task-specific.

### Prompt tuning
- Include additional task-specific “tokens” in the input, then fine-tune only their embeddings for that particular task, while keeping the rest of the model frozen.![[Screenshot 2023-02-20 at 17.29.26.png]]

### Prefix tuning
- In addition to the input, can include these trainable task-specific “tokens” into all layers of the transformer.![[Screenshot 2023-02-20 at 17.29.57.png]]
### Control Prefixes
- Training different prefixes for each property/attribute you want the output to have. For example, the domain or desired length of the text.

### Adapters
- Inserting specific trainable modules into different points of the transformer, while keeping the rest of the model frozen.

### BitFit
- Keep most of the model parameters frozen, fine-tune only the biases. 
- Works surprisingly well, considering it only affects 0.08% of parameters.

### Low-rank adaptation
Keep the main weights frozen but fine-tune two smaller matrices A and B such that the new weights are going to be W’ = W + AB![[Screenshot 2023-02-20 at 17.32.56.png]]


## The keys to good models of language

- Pre-trained transformer models have been a revolution in NLP. 
- After only a couple of years it is difficult to find any model that doesn’t use one of the pre-trained transformer models. 
- This is thanks to a few properties that can be applied to other ML tasks as well, beyond language.

1. Transfer learning (model pre-training)
	-  Pre-training the model gives us better performance even with fewer downstream training examples.
2. Very large models
	- Training bigger models is giving better performance, although diminishing returns.
3. Loads of data
	- They have been trained on exponential magnitude of data
4. Fast computation
	-  In order to process huge amounts of data, the models need to be fast
	- Transformers are not particularly fast. But they are fast for their size.
	- Representations for all the tokens in a sentence can be calculated in parallel. Particularly good for running on GPUs! 
	- In contrast, RNNs and LSTMs would process each word in sequence.
5. A difficult learning task
	- The prediction of missing words (MLM, span-based, etc) is a very difficult task
	- The model can’t just memorise the correct answers.