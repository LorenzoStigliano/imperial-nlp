### Ambiguity at the word level 

Source of complexity for NLP tasks:
1.  Syntactic ambiguity (prepositional phrase attachment ambiguity)
2.  Semantic ambiguity (meaning of a sentence)
3.  Referential ambiguity 
4.  Non-literal meaning (expressions in English that need to be understood)

### Linguistic levels of language understanding

1.  Lexicon - morphological analysis: 
	- Word segmentation
	- Word normalisation (capitalisation, acronyms, spelling variants)
	- Word Lemmatisation -> reduce words into base word: eating, eats -> eat
	- Word stemming -> reduce to root, not always a valid word
	- Byte-pair encoding -> data-driven methods for breaking words apart, used by neural models
	- PoS tagging (Part-of-Speech) what type of word is it (noun, verb,…)
	- Morphological analysis -> recognise/generate word variants

2. Syntax: Sentence structure
	- How are words put together? They are put together by a grammar. Grammar rules (S-> NP VP, …) + Lexicon (words to tag match DOG -> N) we can do PARSING
	- Parsing -> create sentence trees

3.  Semantics: Meaning of words and sentences
	- Word sense disambiguation: given a word and context, assign the meaning given a set of candidates 
	- Compositional meaning: compositing the meaning of the sentence, based on the meaning of the words and the structure of the sentence
	- Semantic role labelling: assign semantic roles to words in sentence

4.  Discourse: Meaning of a text
	- How do sentences relate to one another? Reference and relationships within and across sentences
	- Coreference resolution: mapping different mentions of the same entity together

5.  Pragmatics: Intentions, commands
	- What is the intent behind the text? 

Currently: we learn this with a single model end-to-end, Previously: Learn each individually and join them 

Why ML?
-   Creation and maintenance of language rules is unfeasible to hand write them 
-   Provides flexible learnable framework for representing information 
-   Optimise models end-to-end
-   Can learn supervises and unsupervised data
-   Deep learning -> skip the feature extraction phase, model finds the important features