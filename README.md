# Sentences augmentation

This project enables sentence segmentation using distilbert model for token classification. 
## Usage
Install the necessary packages with this command 
```bash
pip3 install -r requirements.txt
```
Note that you will need **git lfs** to donwload the model weights in the dir models.
To get a list of sentences from a given text, run this command : 
```bash
python3 evaluate.py --fulltext 'your text here'
```
For example , given this query : 
```bash
python3 evaluate.py --fulltext "The first  pig was very lazy he didn't want to work  and he built his house out of straw the second pig worked a little bit harder he was somewhat lazy too he built his house out of sticks. Then, they sang, danced and played together the rest of the day."
```
We get this output : 
```bash
['<span>the first pig was very lazy</span>', "<span>he didn't want to work</span>", '<span>and he built his house out of straw</span>', '<span>the second pig worked a little bit harder</span>', '<span>he was somewhat lazy too</span>', '<span>he built his house out of sticks .</span>', '<span>then , they sang , danced and played together the rest of the day .</span>']
```
Also evaluate.py stores the outputs in a text file. 
## Usecase
Although some famous libraries such as *spacy* contain sentence segmentation algorithms, it is usually limited to the separation of well structured paragraphs , that are separated by connectors such as dots or commas. The model implemented in this repo is also well suited to separate unstructured data , such as the one given in the above example. 
For example , given the text above , spacy gives these sentences : 
```bash

['The first  pig was very lazy',
 "he didn't want to work  ",
 'and he built his house out of straw',
 'the second pig worked a little bit harder he was somewhat lazy too he built his house out of sticks.',
 'Then, they sang, danced and played together the rest of the day.']
 ```
 We can see that it recovered the well defined sentences but mistaked the unclear ones (fourth example).
## Dataset constitution
At first, I used brown dataset from nltk to train the model. However,  many examples of this dataset are composed by more than one sentence. So, I used the method from this [article](https://praneethbedapudi.medium.com/deepcorrection-1-sentence-segmentation-of-unpunctuated-text-a1dbc0db4e98) to generate artificial paragraphs. The idea is to use tatoeba dataset that contains many sentences and concatenate them using some links ( dots , two points , 'and' item ,etc...) randomly. We concatenante at most six sentences per example. 
This method ensures using sentences from broad themes. But it lacks context and some connectors are not well represented in the dataset. For instance , the model can consider two sentences separated by 'but' connector as one. 
More details can be found in the notebook **tatoeba_ds.ipynb**. If you want to skip this step, you can load the training data in data folder.
## Training
I used distillbert model to train the model with transformers library. You can see **distillbert_model_training.ipynb** for details: it includes tokenization step , creating pytorch dataloaders, defining the transformers model and finally training. It contains also the code for evaluation that you can find in evaluate.py. Furthermore, I used colab with gpu to train the model. The training set is limited to 40 000 examples to avoid cuda memory issues. A potential imrovement to the model can be done with more data. 
## Evaluation 
Given a text, we want to extract the sentences that compose it. An important issue that may rise is the text size, since I trained only on paragraphs of 6 sentences at most and bert models can't exceed 512 tokens. So I implemented a " sliding window " method : given the full text , this window will extract at each time a chunk of the data. We evaluate the sentences of this chunk and then we slide the window on the next tokens. One possible problem is if we chunk at the middle of a sentence, so we can have truncated sentences at each window. So what I did is that the chunk i+1 starts at the token just after the last sentence in chunk i. Example  : 
Imagine that we have this text : John is at home and he sleeps. If we use a window of 6 tokens, the first chunk given by the algorithm is John is at home and he. The output sentences will be [John is at home, and he] and the next chunk will be sleeps. Insted , we output only the first sentence , and we consider the second chunk as "and he sleeps".
Sometimes, it is also difficult to recover the original text from distilbert tokens. I added few methods to adjust this but the reconstruction is nevertheless not usually perfect. What we lose is especially capital letters. This can be fixed with some matching algorithms (for instance compare first and last token and take what's between). Also , the algorithm may give errors if we use some special caracters, for instance '-' caracter ( because bert tokenizer consider this as a token by itself which is not usually the case , for example 'ill-tempered').

## Improvements to be made
* Recover the same text. 
* Add some data to improve performance.
* Take into account special caracters.




