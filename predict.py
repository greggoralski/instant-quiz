# -*- coding: utf-8 -*-
"""question generation-multichoice
"""
import nltk
nltk.download('stopwords')
nltk.download('popular')
import streamlit as st

import pprint
import itertools
import re
import pke
import string
from nltk.corpus import stopwords
import spacy
from summarizer import Summarizer
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import requests
import json
import re
import random
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk
from nltk.corpus import wordnet as wn

# def user_input_features():
#     sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
#     sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
#     petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
#     petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
#     data = {'sepal_length': sepal_length,
#             'sepal_width': sepal_width,
#             'petal_length': petal_length,
#             'petal_width': petal_width}
#     # features = pd.DataFrame(data, index=[0])
#     return features

# df = user_input_features()

# st.subheader('User Input parameters')
# st.write(df)
# f = open("egypt.txt","r")
# full_text = f.read()
full_text = "The Nile River fed Egyptian civilization for hundreds of years. It begins near the equator in Africa and flows north to the Mediterranean Sea. A delta is an area near a river’s mouth where the water deposits fine soil called silt. This soil was fertile, which means it was good for growing crops. The red land was the barren desert beyond the fertile region. When the birds arrived, the annual flood waters would soon follow. Then they used a tool called a shaduf to spread the water across the fields. These innovative, or new, techniques gave them more farmland. They were the first to grind wheat into flour and to mix the flour with yeast and water to make dough rise into bread. Egyptians often painted walls white to reflect the blazing heat. Poorer Egyptians simply went to the roof to cool off after sunset. Even during the cool season, chipping minerals out of the rock was miserable work. One ancient painting even shows a man ready to hit a catfish with a wooden hammer. A boomerang is a curved stick that returns to the person who threw it.) The river’s current was slow, so boaters used paddles to go faster when they traveled north with the current. Going south, they raised a sail and let the winds that blew in that direction push them. The Nile provided so well for Egyptians that sometimes they had surpluses, or more goods than they needed. Ancient Egypt had no money, so people exchanged goods that they grew or made. This prosperity made life easier and provided greater opportunities for many Egyptians. For example, some ancient Egyptians learned to be scribes, people whose job was to write and keep records. Some skilled artisans erected stone or brick houses and temples. A few Egyptians traveled to the upper Nile to trade with other Africans. They brought back exotic woods, animal skins, and live beasts. Egyptians created a government that divided the empire into 42 provinces. Many officials worked to keep the provinces running smoothly. Priests followed formal rituals and took care of the temples. Before entering a temple, a priest bathed and put on special linen garments and white sandals. Together, the priests and the ruler held ceremonies to please the gods. By doing so, they hoped to maintain the social and political order. In Egypt, people became slaves if they owed a debt, committed a crime, or were captured in war. Unlike other ancient African cultures, in Egyptian society men and women had fairly equal rights. For example, they could both own and manage their own property. Children in Egypt played with toys such as dolls, animal figures, board games, and marbles. Almost all Egyptians married when they were in their early teens. As in many ancient societies, much of the knowledge of Egypt came about as priests studied the world to find ways to please the gods. Doctors believed that the heart controlled thought and the brain circulated blood, which is the opposite of what is known now. Early Egyptians created a hieroglyphic system with about 700 characters. Legend says a king named Narmer united Upper and Lower Egypt. Some historians think Narmer actually represents several kings who gradually joined the two lands. It combined the red Crown of Lower Egypt with the white Crown of Upper Egypt. When a king died, one of his children usually took his place as ruler. Historians divide ancient Egyptian dynasties into the Old Kingdom, the Middle Kingdom, and the New Kingdom. The Old Kingdom started about 2575 B.C., when the Egyptian empire was gaining strength. In such a case, a rival might drive him from power and start a new dynasty. The first rulers of Egypt were often buried in an underground tomb topped by mud brick. They replaced the mud brick with a small pyramid of brick or stone. It is called a step pyramid because its sides rise in a series of giant steps. He ordered the construction of the largest pyramid ever built. One reason is that the pyramids drew attention to the tombs inside them. Grave robbers broke into the tombs to steal the treasure buried with the pharaohs. Egyptians believed that if a tomb was robbed, the person buried there could not have a happy afterlife. This way, the pharaohs hoped to protect their bodies and treasures from robbers. This was to confuse grave robbers about which passage to take. Tombs were supposed to be the palaces of pharaohs in the afterlife. Mourners filled the tomb with objects ranging from food to furniture that the mummified pharaoh would need. Such activities included growing and preparing food, caring for animals, and building boats. Only a secret tomb built for a New Kingdom pharaoh was ever found with much of its treasure untouched. The dazzling riches found in this tomb show how much wealth the pharaohs spent preparing for the afterlife. This period of Egyptian history is called the Middle Kingdom."

def get_nouns_multipartite(text):
    out=[]
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text)
    #    not contain punctuation marks or stopwords as candidates.
    pos = {'PROPN'}
    #pos = {'VERB', 'ADJ', 'NOUN'}
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    # 4. build the Multipartite graph and rank candidates using random walk,
    #    alpha controls the weight adjustment mechanism, see TopicRank for
    #    threshold/method parameters.
    extractor.candidate_weighting(alpha=1.1,
                                  threshold=0.75,
                                  method='average')
    keyphrases = extractor.get_n_best(n=20)

    for key in keyphrases:
        out.append(key[0])
    return out

def tokenize_sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    # Remove any short sentences less than 20 letters.
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences

# Distractors from Wordnet
def get_distractors_wordnet(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors

def get_wordsense(sent,word):
    word = word.lower()
    if len(word.split())>0:
        word = word.replace(" ","_")
    synsets = wn.synsets(word,'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output =  adapted_lesk(sent, word, pos='n')
        lowest_index = min (synsets.index(wup),synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

# Distractors from http://conceptnet.io/
def get_distractors_conceptnet(word):
    word = word.lower()
    original_word= word
    if (len(word.split())>0):
        word = word.replace(" ","_")
    distractor_list = [] 
    url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
    obj = requests.get(url).json()

    for edge in obj['edges']:
        link = edge['end']['term']
        url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)      
    return distractor_list

def generate_questions():
  model = Summarizer()
  result = model(user_input, min_length=60, max_length = 500 , ratio = 0.4)
  summarized_text = ''.join(result)
  keywords = get_nouns_multipartite(full_text) 
  filtered_keys=[]
  for keyword in keywords:
      if keyword.lower() in summarized_text.lower():
          filtered_keys.append(keyword)
  sentences = tokenize_sentences(summarized_text)
  keyword_sentence_mapping = get_sentences_for_keyword(filtered_keys, sentences)
  key_distractor_list = {}
  for keyword in keyword_sentence_mapping:
      if len(keyword_sentence_mapping[keyword]) > 0:
        wordsense = get_wordsense(keyword_sentence_mapping[keyword][0],keyword)
        if wordsense:
            distractors = get_distractors_wordnet(wordsense,keyword)
            if len(distractors) ==0:
                distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors
        else:
            distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors     
  index = 1
  questions_text = ""
  for each in key_distractor_list:
      sentence = keyword_sentence_mapping[each][0]
      pattern = re.compile(each, re.IGNORECASE)
      output = pattern.sub( " _______ ", sentence)
      st.write ("%s)"%(index),output)
      # questions_text = questions_text + "%s)"%(index),output
      choices = [each.capitalize()] + key_distractor_list[each]
      top4choices = choices[:4]
      st.write ("correct: ", choices[:1])
      # questions_text = questions_text + "correct: ", choices[:1]
      random.shuffle(top4choices)
      optionchoices = ['a','b','c','d']
      for idx,choice in enumerate(top4choices):
          st.write ("\t",optionchoices[idx],")"," ",choice)
          # questions_text = questions_text + "\t",optionchoices[idx],")"," ",choice
      st.write ("\nMore options: ", choices[4:20],"\n\n")
      # questions_text = questions_text + "\nMore options: ", choices[4:20],"\n\n"
      index = index + 1
  # st.write(questions_text)


st.write("""
# Instant Questions
""")
user_input = st.text_area("label goes here", "default_value_goes_here")
if st.button('Generate Questions'):
    generate_questions()

