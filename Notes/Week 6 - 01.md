## Transformers

- Introduced in 2017 
- Very scalable 
- Non-recurrent model, purely based on attention![[Screenshot 2023-02-13 at 16.10.22.png]]
- We have N encoder/decoders with multiple sub layers
![[Screenshot 2023-02-13 at 16.14.07.png]]
## Transformer Encoder
Uses:
- Classification
- Masked language modelling 
- Named entity recognition 
![[Screenshot 2023-02-13 at 16.14.30.png]]
- the Transformer takes in a source sentence, and generates a target![[Screenshot 2023-02-13 at 16.15.03.png]]
- 
![[Screenshot 2023-02-13 at 16.15.03.png]]
- Inside the transformer, we have a stack of encoder layers. 
- We also have a stack of decoder layers 
- At some point during decoding, we will use the encoded output to help us generate the target 
- Note that the output of one encoder layer is sent as input to the next encoder layer. Similarly, the output of one decoder layer is sent to the next decoder layer as input

## Encoder Transformer Architecture 

- INSIDE AN ENCODER![[Screenshot 2023-02-13 at 16.16.15.png]]
- So let’s take a deeper look at the first sublayer in an encoder: The one which contains the multi-head self-attention. 
- We’re going to receive some input - represented here in yellow. The input will be an encoding of each of our words. So here, we have 3 words represented with 4 dimensions. More generally, you would have S words in the input sequence, each represented with D dimensionality. 
- The MHA module processes the input and outputs another set of S x D encodings. - These encodings get sent to the Residual & Norm, which outputs another set of S x D encodings. 
- Conceptually this should make transformers easy to work with - mostly everything in the encoder stays as S x D

### Self Attention
![[Screenshot 2023-02-13 at 16.18.07.png]]
- Each input = each word. So it’s saying that each word looks at every other word (including itself). 
	- Each word therefore gets a representation based on all the other words in the input sequence. 
	- The representation of the sequence is S x D. So we have S words (e.g. The animal didn etc.). Each s in S has looked at every other word to compute its D-dimensional representation![[Screenshot 2023-02-13 at 16.20.00.png]]
	- Transformers are designed such that the hidden state representation of the word “it” would be influenced more by “animal” than “because” (RNN)
![[Screenshot 2023-02-13 at 16.20.31.png]]
1. Define a Q
2. Calculate similarity of Q against the Ks
3. Output is the weighted sum of Vs

Example with a dictionary 
![[Screenshot 2023-02-13 at 16.21.43.png]]![[Screenshot 2023-02-13 at 16.21.55.png]]
Examples of how to join when queries are unseen
![[Screenshot 2023-02-13 at 16.25.12.png]]![[Screenshot 2023-02-13 at 16.25.18.png]]![[Screenshot 2023-02-13 at 16.25.28.png]]

![[Screenshot 2023-02-13 at 16.27.52.png]]
- We learn Q, K, Vs
- So, we have each one of our word encodings… w1, w2, w3. We will call our input sequence X, and the length is S
- We will obtain a query vector for each one of our inputs. So q1 = w1 . $W^Q$; q2 = w2 . WQ etc
- The same thing will happen for our Ks. Each one of our encoded inputs will be sent through WK to produce an S x dh representation. (Again, D = dh here)
- And same for Vs![[Screenshot 2023-02-13 at 16.29.57.png]]
- Q, K, V (SxD)
![[Screenshot 2023-02-13 at 16.33.24.png]]
- We do the above for each word
- Scores is a similarity measure dot product 
- $d_h$ is a large positive number used for division, used to make softmax smoother less peaky, in order so there is no weighting on one word![[Screenshot 2023-02-13 at 16.38.49.png]]
- V is S x $d_h$  so Z is S x $d_h$
- Row-wise softmax is applied to the matrix 

### Multi-head attention 

- Is self-attention but it preforms self-attention head amount of times 
- Intuitively multiple attention heads allows for attending to parts of the sequence differently (e.g. some heads are responsible for longer-term dependencies, others for shorter-term dependencies)
![[Screenshot 2023-02-13 at 17.15.35.png]]
- We get head number of different weights
- D/heads is the $d_h$, where D is the model dimensionality 

## Normalisation 

- Data fed to a neural network should be normalised 
- There are also normalisation techniques within the activations or layers of a network 
	- Batch, Layer, group normalisation![[Screenshot 2023-02-13 at 17.26.33.png]]
![[Screenshot 2023-02-13 at 17.28.54.png]]![[Screenshot 2023-02-13 at 17.29.28.png]]
- By adding the previous layer's output directly to the current layer's output, a residual connection allows the current layer to focus on learning the difference between the two outputs, rather than learning an entirely new transformation.

## Position-wise Feedfoward Network 
![[Screenshot 2023-02-13 at 17.34.07.png]]
- A position-wise FFN is an MLP
- Position-wise just means it is applying the same transformation to every element in the sequence ![[Screenshot 2023-02-13 at 17.33.31.png]]
This is the encoder all together we have N of these 
![[Screenshot 2023-02-13 at 17.35.51.png]]
- Our diagram indicates that the output of one layer forms the input to the next 
- Transformers are position invariant by default
- Positional encodings are a way to inject position information into the embeddings

## Positional Encodings
![[Screenshot 2023-02-13 at 17.38.25.png]]
- The positional encoding vector is a small vector which is added to the embedding of x. 
- The resulting vector will be in the same general space as the original embedding 
- The function which gets the positional encoding is purely a function of the timestep of x.
- Positional Encoding does not depend on the features of any given word. Only the position that it appears in. e.g. index 1, index 2, index 3 etc![[Screenshot 2023-02-13 at 17.40.36.png]]
- ![[Screenshot 2023-02-13 at 17.40.48.png]]
- i is an index 
- Early on exponent will be small and so the sin and cos functions will be large
- Later on it will be small![[Screenshot 2023-02-13 at 17.43.05.png]]