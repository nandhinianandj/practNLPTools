# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name :
#
#* Purpose :
#
#* Creation Date : 21-11-2020
#
#* Last Modified : Monday 23 November 2020 10:17:40 PM IST
#
#* Created By : Nandhini Anand Jeyahar

#_._._._._._._._._._._._._._._._._._._._._.#

import json
import neuralcoref
import requests
import spacy

###### Get labelling code
def get_tags(text):
    errorFlag = False
    tags = None
    try:
        url = "http://127.0.0.1:9090/process"
        payload = {"sent": text}
        headers = {'content-type': "application/json"}
        response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
        tags = json.loads(response.text.encode('utf8'))['response']
    except:
        tags = None
    return tags

def ner_triplet(answer):
    url = "http://127.0.0.1:9092/get_triplet"
    payload = {"query": answer}
    headers = {'content-type': "application/json"}
    response = requests.request(
        "POST", url, data=json.dumps(payload), headers=headers)
    tags2 = json.loads(response.text.encode('utf8'))
    return tags2

####Code for anaphora resolution ##########
def neural_coref_resolution(text):
    nlp = spacy.load('en_core_web_lg')
    coref = neuralcoref.NeuralCoref(nlp.vocab)
    nlp.add_pipe(coref, name='neuralcoref')
    text = text.lower()
    doc = nlp(text)
    return doc

def tokenize_text(text):
    token_sen = nltk.sent_tokenize(text)
    word = []
    for i in range(len(token_sen)):
        word.append(nltk.word_tokenize(token_sen[i]))
    return word

##list prps and nn
pronouns = ['PRP', 'PRP$']
nouns = ['NNP', 'NN', 'NNS', 'NNPS', 'JJ', 'JJR']

def splitString(str):
    alpha = ""
    num = ""
    for i in range(len(str)):
        if (str[i].isdigit()):
            num = num + str[i]
        elif ((str[i] >= 'A' and str[i] <= 'Z') or
              (str[i] >= 'a' and str[i] <= 'z')):
            alpha += str[i]
    return num + ' ' + alpha


def coref_rephrase(text, coref=None):
    check_alphanum = re.findall("[a-zA-Z]+[0-9]+|[0-9]+[a-zA-Z]+", text)

    if len(check_alphanum) > 0:
        for x in check_alphanum:
            split_alphnum = splitString(x)
            text = text.replace(x, split_alphnum)
    text = text.encode('ascii', 'ignore').decode('ascii')  ##
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    if not coref:
        coref = coref_resolution(text)
    process_text = tokenize_text(text)
    for coref_entity in coref:
        for coref_entity_element in coref_entity:

            pos_tag_left = nltk.pos_tag([coref_entity_element[0][0]])
            pos_tag_right = nltk.pos_tag([coref_entity_element[1][0]])

            if pos_tag_left[0][1] in pronouns and pos_tag_right[0][1] in nouns:
                if pos_tag_left[0][0] in process_text[coref_entity_element[0][1]]:
                    tmp_flag = False
                    if len(str(pos_tag_right[0][0]).split(' ')) > 2:
                        doc = nlp(str(pos_tag_right[0][0]))
                        spacy_NER = {}
                        for ent in doc.ents:
                            spacy_NER[ent.text] = ent.label_
                        if len(spacy_NER) == 1:
                            for key, value in spacy_NER.items():
                                if value in ['PERSON', 'ORG']:
                                    process_text[coref_entity_element[0][1]][
                                        process_text[coref_entity_element[0][1]].index(pos_tag_left[0][0])] = \
                                    process_text[coref_entity_element[0][1]][
                                        process_text[coref_entity_element[0][1]].index(pos_tag_left[0][0])] + '0#' + str(
                                        key) + '1#'
                                    tmp_flag = True
                                    break
                        elif len(spacy_NER) > 1:
                            for key, value in spacy_NER.items():
                                if process_text[coref_entity_element[0][1]][process_text[coref_entity_element[0][1]].index(pos_tag_left[0][0])].lower() in ['it','its',"it's"]:
                                    if value in ['ORG']:
                                        process_text[coref_entity_element[0][1]][
                                            process_text[coref_entity_element[0][1]].index(pos_tag_left[0][0])] = \
                                        process_text[coref_entity_element[0][1]][
                                            process_text[coref_entity_element[0][1]].index(pos_tag_left[0][0])] + '0#' + str(
                                            key) + '1#'
                                        tmp_flag = True
                                        break
                                elif value in ['PERSON']:
                                    process_text[coref_entity_element[0][1]][
                                        process_text[coref_entity_element[0][1]].index(pos_tag_left[0][0])] = \
                                    process_text[coref_entity_element[0][1]][
                                        process_text[coref_entity_element[0][1]].index(pos_tag_left[0][0])] + '0#' + str(
                                        key) + '1#'
                                    tmp_flag = True
                                    break

                    if tmp_flag is False and len(pos_tag_right[0][0].split(' ')) < 6 :
                        process_text[coref_entity_element[0][1]][
                            process_text[coref_entity_element[0][1]].index(pos_tag_left[0][0])] = \
                        process_text[coref_entity_element[0][1]][
                            process_text[coref_entity_element[0][1]].index(pos_tag_left[0][0])] + '0#' + str(
                            pos_tag_right[0][0]) + '1#'

    rephrase = [' '.join(word) for word in process_text]
    return rephrase


