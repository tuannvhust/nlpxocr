import py_vncorenlp
from transformers import AutoModel, AutoTokenizer

from transformers import MBartForConditionalGeneration
from utlis import get_args
import json
# import models
import activation
# from models import Xattention
# from activation import SoftMax
#BARTpho-syllable
import numpy
import math
from distance import levenshtein_distance
from score import score

uni_dictionary = []
# Opening JSON file
f = open('vn_syllables.json',)
data = json.load(f)
for i in data:
    uni_dictionary.append(i)

# Closing file
f.close()

def prediction(txt,label):
    txt = txt.lower()
    prediction_mask = []
    candidates = [txt]
    txt_score = score(txt)
    candidates_score = [txt_score]
    for j in range(len(txt.split())):
        word_list = txt.split()
        for i in uni_dictionary:
            if levenshtein_distance(word_list[j],i)==1: #or levenshtein_distance(j,i)==2:
                prediction_mask.append(i.lower())
        prediction_mask.append(word_list[j])
        prediction_mask = [*set(prediction_mask)]
        for k in prediction_mask:
            new_word_list = txt.split()
            new_word_list[j] = k
            new_sentence = " ".join(new_word_list)
            score_new_sentence = score(new_sentence)
            candidates.append(new_sentence)
            candidates_score.append(score_new_sentence)
            min_value = min(candidates_score)
            min_index = candidates_score.index(min_value)
            best_sentence = candidates[min_index]
    return label + "|" + best_sentence

def main(args):
    with open(args.test_file,'r') as test_file:
        test_set = [line.strip('\n') for line in test_file]
        for i in test_set:
            ocr_prediction = i.split('|')[0]
            ocr_label = i.split('|')[-1]
            a = prediction(ocr_prediction,ocr_label)
            with open('metadata.txt','a') as file:
                file.writelines(a + '\n')
            # mask_dict = replace_word_with_mask(ocr_prediction,ocr_label)
            # new_mask_dict = []
            # for j in range(len(ocr_prediction.split(" "))):
            #     prediction_mask = prediction(mask_dict[ocr_label][j])
            #     bartpho_prediction_dict = replace_mask_with_bartpho_prediction(prediction_mask,mask_dict[ocr_label][j],ocr_label,new_mask_dict)
            # print(bartpho_prediction_dict)




def replace_word_with_mask(sentence,label):
    words = sentence.split()
    masked_strings = []
    for i in range(len(words)):
        masked_words = words.copy()
        masked_words[i] = "<mask>"
        masked_string = " ".join(masked_words)
        masked_strings.append(masked_string)
    
    output_dictionary = {label: masked_strings}
    return output_dictionary


def replace_mask_with_bartpho_prediction(prediction_mask, mask_sentence,ocr_label,new_mask_dict):
    mask_strings = new_mask_dict
    for i in prediction_mask:
        new_mask_sentence = mask_sentence.replace('<mask>',i)
        mask_strings.append(new_mask_sentence)
    output_directory = {ocr_label:mask_strings}
    return output_directory

# def get_embedding_model(args):
#     model = {
#         'attention': Xattention(args).to(args.device),
#     }
#     try:
#         return model[args.embedding_model]
#     except:
#         NotImplementedError

# def get_model(args):
#     model = {
#         'softmax' : SoftMax(get_embedding_model(args),args).to(args.device),
#         }
#     try:
#         return model[args.metric]
#     except:
#         NotImplementedError



if __name__=="__main__":
    args = get_args()
    main(args)

    # print(score_return)