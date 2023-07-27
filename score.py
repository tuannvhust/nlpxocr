# from transformers import BertTokenizer, BertForMaskedLM
# from transformers import AutoModel, AutoTokenizer
import torch
import math
from transformers import pipeline
nlp = pipeline("fill-mask", model="vinai/phobert-large")

# model = AutoModel.from_pretrained("vinai/phobert-large")
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
def score(sentence):
    # tokenizer = BertTokenizer.from_pretrained('vinai/phobert-base')
    # model = BertForMaskedLM.from_pretrained('vinai/phobert-base')
    # model.eval()
    sentence_list = sentence.split()
    score= []
    for i in range(len(sentence_list)):
        # init softmax to get probabilities later on
        sm = torch.nn.Softmax(dim=0)
        torch.set_grad_enabled(False)
        mask = nlp.tokenizer.mask_token
        new_sentence_list = sentence.split()
        new_sentence_list[i] = mask
        new_sentence = ' '.join(new_sentence_list)
        # token_ids = torch.tensor([tokenizer.encode(new_sentence)])
        # # token_ids = tokenizer.encode(sentence, return_tensors='pt',padding=True,truncation=True)
        # # print(token_ids)
        # # tensor([[ 101, 1045,  103, 2017,  102]])
        # # get the position of the masked token
        # masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()

        # # forward
        # output = model(token_ids)
        # last_hidden_state = output[0].squeeze(0)
        # # only get output for masked token
        # # output is the size of the vocabulary
        # mask_hidden_state = last_hidden_state[masked_position]
        # # convert to probabilities (softmax)
        # # giving a probability for each item in the vocabulary
        # probs = sm(mask_hidden_state)
        # prob_shape = [*probs.shape]

        # # get probability of token 'hate'
        # hate_id = tokenizer.convert_tokens_to_ids(sentence_list[i])

        # hate_prob = probs[hate_id].item()

        # print(hate_prob)
        a = nlp(new_sentence, targets=new_sentence_list[i])
        hate_prob = a[0]['score']

        
        score.append(hate_prob)
    score_return = math.log(sum(score))
    return score_return