# GPTfromScratch
Following Andrej Karpathy's Let's build GPT youtube video. This project builds out the decoder section of the transformer model and trains it
on a dataset of all of Shakespeare's work. The result generates new Shakespeare-like stories!

Since this is only for pedagogical purposes, I have annotated the gpt-dev.ipynb and bigramv2.py with my own comments, and added this ReadMe to explain high level concepts. input.txt is the training dataset. bigramv2.py is the final script that builds the encoder and produces tiny shakespeare.

Let's start by understanding the architecture of the Transformer. In the 2017 paper "Attention is all you need", the authors were attempting to solve a machine translation problem. For example, having a machine translate French to English. There are two main parts to the architecture, the encoder and the decoder. The encoder takes in the French sentence "Les réseaux de neurones sont géniaux!" and conditions it, and the decoder is expected to generate the response "<START> Neural networks are awesome!<END>". For our purposes, we only need the decoder to create Shakespeare stories since there is no prompting. For the author's purposes, they want the decoder's response to be dependent on the encoder (sentence to be translated). The diagram below shows the whole transformer model with red X's removing the encoder.
 

![alt text](<transformer only decoder.JPG>)

The key innovation that has allowed the transformer model to take over AI is Attention. To understand why, let's first understand why historical context (words that came before the word we want to generate) is important. 

Take the following example: if we have the incomplete sentence "Tom" and I asked you to generate the next word, a reasonable response could be "Cruise". If instead you were given more context to the same sentence, for example "In 2021, the Superbowl MVP quarterback was Tom", you would now give a different response "Brady". If you were given a different sentence altogether like "The singer of Last Dance with Mary Jane is Tom", your answer is now "Petty". You only know the correct answer because you were given the full context before being asked to give the final word. Generally speaking, to generate better answers we want to gather information from the past, but we want to do it in a data dependent way. This is the problem Attention seeks to solve.

Attention solves this by introducing keys and queries. Each word in our sentence is given a key and a query. Roughly speaking, the key tells us what 'data' the word has, and the query tells us what 'data' the word is looking for. The name "Attention" comes from the idea that each word is paying attention to how well its key matches to another word's query. When a key and query line up well together (their dot product is high), the model will get to learn more about those words compared to others. Going back to the example, the word "Brady" will be trained to have a query that aligns very well with the keys of "Superbowl", "MVP", and "quarterback" while the word "Petty" will align with "singer" and "Mary Jane". When we do this for all the words in the sentence, we get a matrix that tells us how aligned all the words are with each other! Let's call this the weight matrix. The last piece of Attention is the values, which every word has just like a key and query. The value is the piece of information that gets communicated from one word to another, and is scaled by the weight matrix index that corresponds those two words together. 

This whole key, query, and value process is called a 'Head' of Attention. Simply put, the input to this Head of Attention are the words you want to pay 'attention' to each other, and the output is the weighted sum of the values, where the weight is a measure of the word compatability. Put more technically, the input is a (Batch, Time, Channel) tensor and the output is a (Batch, Time, Head Size) tensor of the computed weighted sum of values.

Some notes on attention
1. At its core, attention is a communication mechanism. It provides a way for nodes in a graph to look at each other and interact more or less depending on their affinities (rated by their dot product of key and query)

2. There is no notion of space in attention. It's simply a mathematical act over some vectors. To add this functionality, we need to positionally encode tokens to say 'hey, you are token number 5.'

3. Sequences that are in different batches DO NOT talk to eachother. They are processed independently.

4. The idea that 'next token generation cannot have inputs from future tokens' is not always true. There are some cases where you want all the tokens to communicate, such as sentiment analysis. For this, just delete the masked_fill line.

5. Self-attention is when your nodes all have their own values, keys, and queries, and they all talk to eachother. Attention is more general than that. A different example is cross-attention, where your nodes have values, but the keys and queries come from a different set.

6. We need to divide our wei by 1 over square root of head_size. If you don't, the variance of wei will be on the order of head_size, which means that softmax will be very skewed towards the extremes. By including this factor, the var stays around 1, making softmax work much better.



