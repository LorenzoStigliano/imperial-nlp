- Differences between encoder and decoder
	- In decoder there is a cross-attention mechanism ![[Screenshot 2023-02-16 at 11.14.02.png]]
- Decoder overview:![[Screenshot 2023-02-16 at 11.14.14.png]]
	- Masked multi-head self-attention 
	- Cross attention. Note that cross attention is also receiving a key and value tensor. These key and value tensors come from the encoder. We’ll look at this more in a future slide

## Training vs Testing 

### Testing
![[Screenshot 2023-02-16 at 11.15.28.png]]
- Namely, we perform auto-regressive generation 
	1. Our source sentence gets encoded via our encoder
	2. We feed an SOS token to our decoder. It then predicts the first word (i.e. I) 
	3. We append the prediction to our SOS token, and then use this to predict the next word (i.e. am) 
	4. And so forth. At some point our decoder will predict an EOS token (not shown here), and we’ll know that this is the end of the generation.

### Training 
![[Screenshot 2023-02-16 at 11.16.18.png]]
- However, during training, we won’t get the model to generate auto-regressively. 
- We’re actually going to to feed the whole target sequence as input to the decoder and decode the sentence in one go
- This means we can run the decoder stack once.
- As opposed to running it for as many tokens as we have in the target sequence (which is what we do when performing inference)
- If we feed in the whole sequence to the decoder, we need a way to tell the decoder not to look at future tokens when computing attention for one of the tokens.
	- We enforce this unidirectionality with masking

## Masked Multi-head Self-attention
- When decoding we do not have access to future tokens
- Masked MHA is a strategy to tell the model during training to not look at future tokens
- We’re going to enforce unidirectionality in the self-attention mechanism by using a mask.
- Here, unidirectionality means that we should only be looking at words to the left.
- Mask is applied on top of the attention matrix, upper triangular matrix to a very large negative number ![[Screenshot 2023-02-16 at 11.18.57.png]]

## Cross Attention
- The decoder needs to know the encoded representation
- Every time we are decoding a token, we need to know which encoded words we should look at to decode the token 
	- Achieved with cross attention
- We use the key, value tensors from the **last encoder layer** and send them to **all** the decoder layers
	- Query comes from the current decoder layer
- Cross attention matrix shape = T x S

## Transformers
![[Screenshot 2023-02-16 at 11.42.18.png]]
Other tricks to train
![[Screenshot 2023-02-16 at 11.42.39.png]]