# Copied from https://medium.com/analytics-vidhya/question-answering-system-with-bert-ebe1130f8def

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

#Model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

paragraph = None
question = None

with open("data/document.txt") as f:
    paragraph = f.read()

with open("data/query.txt") as f:
    question = f.read()
            
encoding = tokenizer.encode_plus(text=question,text_pair=paragraph)

inputs = encoding['input_ids']  #Token embeddings
sentence_embedding = encoding['token_type_ids']  #Segment embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens

result = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
start_scores = result.start_logits
end_scores = result.end_logits

start_index = torch.argmax(start_scores)

end_index = torch.argmax(end_scores)

answer = ' '.join(tokens[start_index:end_index+1])

corrected_answer = ''

for word in answer.split():
    
    #If it's a subword token
    if word[0:2] == '##':
        corrected_answer += word[2:]
    else:
        corrected_answer += ' ' + word

if not corrected_answer or any(['[CLS]' in corrected_answer, '[SEP]' in corrected_answer]):
    print("I don't know")
else:
    print(corrected_answer)