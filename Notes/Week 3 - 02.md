

## Classification

### What is a bias? When is a feature a bias?
 - Researchers may have different perspectives 
 - Ultimately, it’s empirical… whatever helps our model generalise 
 - Is ‘bias’ the right word? We’re specifically targeting features that harm generalisability

### De-biasing 

- Preventing a model learning from shallow heuristics
- How can we make models generalise better?

#### Possible strategies

- Augment with more data so it balances the bias
- Filter data - but loose valuable data
- Make your model predictions different from predictions based on the bias
- Prevent a classifier finding the bias in your model representations

Lets look at point 3:
- The probabilities of each class given our bias ($b_i$).
	- I will call these the “bias probabilities”, probability of class given only the bias.
 - and the probabilities from our model ($p_i$)![[Screenshot 2023-01-30 at 10.05.30.png]]
- Here we learn the bias and then in inference we take away $b_i$ so we have no bias in inference.

#### How to get bias probabilities

- Target a specific known biased feature

We can create a biased model:
1. Use a model not powerful enough for the task; TinyBERT model for shallow heuristics
2. Use incomplete information 
3. Train a classifier on a very small number of observations
4. weight the loss of examples based on the performance of our biased model: 
	- When training the robust model, multiply the loss by: $1-b_i$, where this is the bias probability for the correct class, this is a down weight where bias influences more

When is it desirable to stop a model learning from shallow-heuristics in the dataset?
![[Screenshot 2023-01-30 at 10.23.42.png]]

### Evaluation metrics for classification
![[Screenshot 2023-01-30 at 16.09.21.png]]
![[Screenshot 2023-01-30 at 16.09.40.png]]![[Screenshot 2023-01-30 at 16.10.05.png]]
![[Screenshot 2023-01-30 at 16.10.21.png]]
