import numpy as np
import torch
import transformers
import os
import csv

class SentenceFetch(object):
  
  def __init__(self, data):
    self.data = data
    self.sentences = []
    self.tags = []
    self.sent = []
    self.tag = []
    
    # make tsv file readable
    with open(self.data) as tsv_f:
      reader = csv.reader(tsv_f, delimiter='\t')
      
      for row in reader:
        if len(row) == 0:
          if len(self.sent) != len(self.tag):
            break
          self.sentences.append(self.sent)
          self.tags.append(self.tag)
          self.sent = []
          self.tag = []
        else:
          self.sent.append(row[0])
          self.tag.append(row[1])   

  def getSentences(self):
    return self.sentences
  
  def getTags(self):
    return self.tags

def extractFromDirectories(parent_dir, corpus_path, file_type = 'train.tsv'):
    sentences, tags = [], []

    if os.path.exists(os.path.join(parent_dir, corpus_path)):
        files = os.listdir(os.path.join(parent_dir, corpus_path))
        for file in files:
            if file == file_type:
                
                current_path = os.path.join(parent_dir, corpus_path, file)
                sentence = SentenceFetch(current_path).getSentences()
                tag = SentenceFetch(current_path).getTags()
                sentences.extend(sentence)
                tags.extend(tag)
    else:
        raise FileNotFoundError(f"The folder {corpus_path} was not found in the parent directory.")
    
    print('Number of samples: ',len(sentences))
    
    return sentences, tags

def encode_sentence(sentence, tokenizer):
    #simply encode sentence using encode_plus
    encoded_sent = tokenizer.encode_plus(sentence,
                                         is_split_into_words=True,
                                         truncation=True,
                                         padding='max_length',
                                         max_length=50,
                                         return_token_type_ids=False,
                                         add_special_tokens=False,
                                         )
    return encoded_sent

def align_tags(encoded_sent, tags, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(encoded_sent['input_ids'])
    aligned_tags = []
    counter = 0
    for idx, token in enumerate(tokens):
        #check if its a sub-word
        if token.startswith("##"):
            #check previous tag value and copy that
            if tags[counter] == 'O':
                aligned_tags.append('O')
            #if its an entity, then it can't be the beginning so append I-Entity
            else:
                aligned_tags.append('I-Protein')
        
        elif token == '[PAD]':
            aligned_tags.append('[PAD]')
        else:
            aligned_tags.append(tags[counter])
            #only increment counter if valid token is found
            counter+=1
    
    return aligned_tags

def map_tags(alinged_tags, mapping=None):
    if not mapping:
        label2id = {'O':1, 'B-Protein':2, 'I-Protein':3, '[PAD]':4}
    else:
        label2id = mapping
    def map_labels(label):
        return label2id[label]
    

    mapped_labels = np.vectorize(map_labels)(alinged_tags)

    return list(mapped_labels)

def merge_tags(encoded_sent, aligned_tags):
    #add tags to encoded_sentence dict
    if len(encoded_sent['input_ids']) == len(aligned_tags):
        encoded_sent['labels'] = aligned_tags
    else:
        raise ValueError("Lengths of sentences and tags do not match.")
    
    return encoded_sent

def get_faulty_sentences(sentences):
    delimiter = '\t'
    #new_sentence = []
    faulty_sentences = []
    counter = 0
    for idx, sentence in enumerate(sentences):
        for word in sentence:
            if delimiter in word:
                faulty_sentences.append(idx)
    
    return set(faulty_sentences)
    

def tokenize_dataset(sentences, tags, tokenizer):
    tokenized_dataset = []
    faulty_set = get_faulty_sentences(sentences)
    
    for idx, sentence in enumerate(sentences):
        if idx not in faulty_set:
            encoded_sent = encode_sentence(sentence, tokenizer)
            aligned_tags = align_tags(encoded_sent, tags[idx], tokenizer)
            mapped_labels = map_tags(aligned_tags)
            final_encoding = merge_tags(encoded_sent, mapped_labels)
            tokenized_dataset.append(final_encoding)
            
    return tokenized_dataset

def save_tokenized_data(file_path, data):
    with open(file_path, 'wb') as outfile:
        pickle.dump(data,outfile)

def save_as_tensor_dataset(file_path, data):
    #extract input_ids, attention_masks and labels from list of data
    input_ids = torch.tensor([d['input_ids'] for d in data])
    attention_masks = torch.tensor([d['attention_mask'] for d in data])
    labels = torch.tensor([d['labels'] for d in data])
    dataset = torch.utils.data.TensorDataset(input_ids, attention_masks,labels)

    #save dataset to disk
    torch.save(dataset, file_path)
     
