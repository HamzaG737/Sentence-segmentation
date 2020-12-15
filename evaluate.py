
import numpy as np
import torch
import transformers as ppb
import warnings
warnings.filterwarnings("ignore")
import pickle
import nltk
import argparse
nltk.download('punkt')

parser = argparse.ArgumentParser(description='Glose challenge parser')
parser.add_argument('--fulltext', type=str, metavar='f',
                    help="the text to segment")

parser.add_argument('--window_length', type=int, metavar='w', default=10)
parser.add_argument('--output_path', type=str, metavar='o', default='data/sentences.txt')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer_class, pretrained_weights = (ppb.DistilBertTokenizerFast,'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights,do_lower_case=True)

model = ppb.DistilBertForTokenClassification.from_pretrained(
    pretrained_weights, # Use the distillbert model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.to(device)
PATH = 'models/distillbert_model_glose_finetuned_3.pth'
model.load_state_dict(torch.load(PATH,map_location=device))

def match_sent(sentences , output_path):
  """
  The reconstruction of sentences with distilbert tokenization to recover 
  the original sentence can't be done with just "join". So we add some preprocessing
  steps to recover approximately the same sentences. Note that we will lose some
  properties such as capital letters. 
  This function adds also spans + save all sentences in a text file.
  """
  sentences = [sentence.replace(' ##','') for sentence in sentences]
  sentences = [sentence.replace(" ' ","'") for sentence in sentences]
  sentences = [sentence.replace("did n't","didn't") for sentence in sentences]

  ## add span 
  sentences = ["<span>"+sentence+'</span>' for sentence in sentences]
  with open(output_path, 'w') as output:
    for sentence in sentences:
        output.write(sentence + '\n')
  return sentences
def normalize (preds): 
  """
  fonction that replaces 11 (i.e two adjacent tokens that both represent the ending of a sentence) 
  with 10 to avoid errors.
  """
  l = list(preds)
  string_list = ''.join(map(str,l))
  string_list = string_list.replace('11', '01')
  new_preds = np.array(list(map(int, list(string_list))))
  return new_preds
def get_sentences (indexes_end,tokens_recov,sentences) :
  """
  given the indexes of tokens that end sentences and the list of all the tokens ,
  This function gives the list of all sentences contained in window_sentences.
  """
  current = []
  for k in range(len(tokens_recov)) :
    current.append(tokens_recov[k])
    if k in indexes_end :
      sentences.append(" ".join(current))
      current = []
  return sentences

full_text = args.fulltext
tokenized_text = nltk.word_tokenize(full_text.lower()) ## tokenize all text with nltk
model.eval()

sentences = []
max_length = args.window_length ## size of sliding window
current_begin = 0 ## beginning index of window_sentences , relative to tokenized_text.
moving_add = 0 ## we will use this if window_sentences is an unfinished sentence.
window_sentences = tokenized_text[:max_length]
j,t=0,0
while len(window_sentences) !=0 : 
  j+=1
  inputs_enc = tokenizer(window_sentences, is_split_into_words= True, return_offsets_mapping=False, 
                       padding=False, truncation=True)
  with torch.no_grad():     
    input_ids_ = torch.tensor(inputs_enc.input_ids).unsqueeze(0).to(device)
    outputs = model(input_ids_)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
  preds = np.argmax(logits, axis=2).flatten()[1:-1] ## take all except cls and sep preds
  preds = normalize(preds)
  tokens_recov = tokenizer.convert_ids_to_tokens(inputs_enc['input_ids'])[1:-1]
  
  ## get the indexes of elements that end sentences
  indexes_end = np.where(preds==1)[0]
  sentences = get_sentences (indexes_end,tokens_recov ,sentences)

  if len(indexes_end)==1 : # if we have only one ending token , in the end of the sentence
      ## this case means that there is no ending token except the default last one, 
      ## so we add 10 tokens to sentences test
      moving_add +=10  

      ## we stop if we exceed tokenized_text twice.
      if current_begin+max_length+moving_add>len(tokenized_text): 
        t+=1
        if t == 2 : 
          break 
      window_sentences = tokenized_text[current_begin:current_begin+max_length+moving_add]
      sentences.pop(-1)
      continue
      
      #current_begin += max_length

  moving_add=0
  
  ## this is in case we hove more than two ending tokens. 
  last_sent = sentences[-1] # we will remove last sentence.
  first_token = sentences[-1].split()[0]
  
  indexes_first = np.where(np.array(window_sentences) == first_token)[0]
  if len(indexes_first)>1 : 
    for index in reversed(list(indexes_first)) : 
      if index<=(len(window_sentences)-len(sentences[-1].split())+4) :
        index_first = index
        break

  else :
    index_first=indexes_first[0]
  ## window_sentences will be defined as the window beginning from the last sentence and we add max_length tokens
  window_sentences = tokenized_text[current_begin+index_first:current_begin+index_first+max_length]
  if current_begin+index_first > len(tokenized_text) : 
    break
  sentences.pop(-1)
  current_begin += index_first
sentences = match_sent(sentences,args.output_path)
print(sentences)