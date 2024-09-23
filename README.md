# GPTfromScratch
Following Andrej Karpathy's Let's build GPT youtube video. This project builds out the decoder section of the transformer model and trains it
on a dataset of all of Shakespeare's work. The result produces new Shakespeare like text/stories!
The encoder is not used. See diagram below that shows this. 

![alt text](<transformer only decoder.JPG>)

Since this is only for pedagogical purposes, I have annotated the gpt-dev.ipynb and bigramv2.py with my own comments, and added this ReadMe to explain high level concepts. input.txt is the training dataset. bigramv2.py is the final script that builds the encoder and produces tiny shakespeare!

The tokenizer used simply converts individual characters to integers. A Byte Pair Encoding tokenizer is in the works =). 