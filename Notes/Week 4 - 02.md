### 1. Feed-Forward Neural Language Models

- Neural-based LMs have several improvements:
	- Avoids n-gram sparsity issue 
	- Contextual word representations i.e. embeddings  
- FFLM quickly superseded by RNN LMs![[Screenshot 2023-02-03 at 15.55.27.png]]
- First applications of neural networks to LM 
	- Approximates history with the last C words 
	- C affects the model size
- 4-gram FFLM has a context size of 3 
	- The context is formed by concatenating word embeddings
- Quickly superseded by RNN LMs

### 2. RNNs
![[Screenshot 2023-02-03 at 15.57.49.png]]

Problems: Vanishing gradients
- Why do our nonlinear activation functions contribute?
	- Sigmoid? Derivative is bounded by 0 and 0.25 and the gradient when propagating backwards will get smaller
	- Tanh? The same as before
- What else contributes to vanishing gradients / exploding gradients?
	- We have repeated W matrix multiplication 

Vanilla RNNs: Many-to-many 
- Every input has a label:
	- Language modelling -> predicting the next word 
- The LM loss is predicted from the cross-entropy losses for each predicted word![[Screenshot 2023-02-03 at 16.08.45.png]]
- We look at the log probabilities because we use the cross entropy loss where $q(x_i)$ is the probability of the model given the word

Teacher forcing - we forced on what the model should see not what the model predicts example of what could happen without teacher forcing, so this error could propagate through: ![[Screenshot 2023-02-03 at 16.14.21.png]]

Weight tying, reducing the no. parameters - We can use the same embedding weights in our output layer:![[Screenshot 2023-02-03 at 16.14.56.png]]
The embedding layer E maps our one-hot-label of the input into a word embedding:
- $E$ has dimensions: H x |V| 
- $E^T$ therefore has dimensions: |V| x H

### Bi-directional RNNs
![[Screenshot 2023-02-03 at 16.21.22.png]]
- For classification: We can concatenate the representations at the end of the RNNs for both directions for sentence level classification for the whole sentence, since this gives us all of the information from both RNNs![[Screenshot 2023-02-03 at 16.24.24.png]]
- Multi-layered RNNs![[Screenshot 2023-02-03 at 16.25.34.png]]
	- We can feed in our hidden state from an earlier later into the next layer
- Multi-layered RNNs and bidirectional ![[Screenshot 2023-02-03 at 16.27.45.png]]
### 3. LSTM
![[Screenshot 2023-02-03 at 16.37.30.png]]
- Cell states ($C_t$) represent ‘long term memory’ 
- Hidden states ($h_t$) is current working memory (e.g. the same as for vanilla RNN)
How they work: 
![[Screenshot 2023-02-03 at 16.38.42.png]]
![[Screenshot 2023-02-03 at 16.39.53.png]]
- Element wise multiplication, $f_t$ is H dimensional (hidden layer)
- The weights used in the Forget, Input and Output gates are DIFFERENT parameters
![[Screenshot 2023-02-03 at 16.40.18.png]]
![[Screenshot 2023-02-03 at 16.40.46.png]]
![[Screenshot 2023-02-03 at 16.41.35.png]]
![[Screenshot 2023-02-03 at 16.42.00.png]]
- The output gate and $tanh$ is what differentiates $h_t$ and $c_t$

#### Why do LSTMs help with vanishing gradients?
Consider our cell state (long range memory storage): ![[Screenshot 2023-02-03 at 16.39.53.png]]

What helps?
- The gradients through the cell states are hard to vanish

Two reasons why: 
1. Additive formula means we don’t have repeated multiplication of the same matrix (we have a derivative that’s more ‘well behaved’) 
2. The forget gate means that our model can learn when to let the gradients vanish, and when to preserve them. This gate can take different values at different time steps.

A simplified architecture (GRU)
![[Screenshot 2023-02-03 at 16.45.22.png]]

 