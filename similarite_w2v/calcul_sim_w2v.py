# Calcul de la similarité Word2vec avec le lexique
# Date de création : 25/04/2022
# Dernière MàJ : 30/04/2022
# Chloé Choquet

# Imports des librairies et ressources
import time
time0 = time.time()
import spacy
nlp = spacy.load('fr_core_news_md', exclude=["tagger", "parser", "lemmatizer","ner","textcat"])
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim
import numpy as np
from scipy import spatial
from tqdm import tqdm

# Chemins et variables
w2v_path = './Ressources_w2v/fr/fr.bin'
model = gensim.models.Word2Vec.load(w2v_path)
chemin_lexique = "lexique.csv"
chemin_corpus = "articles_bf.csv"
chemin_sortie = "articles_bf_calcul_w2v.csv"

# Placer les termes du lexique dans des listes
lex_general = []
lex_gen = ""
lex_secu_alim = []
lex_crise = []
with open(chemin_lexique, "r", encoding="utf8") as fic_lexique :
    contenu_lexique = fic_lexique.readlines()
for i in tqdm(range (1, len(contenu_lexique)), desc="Extraction des lexiques : ") : # on commence à l'indice 1 pour ne pas avoir la ligne de titre ("LEXG","LEXA", "LEXC")
    ligne = contenu_lexique[i].split("\t")
    if len(ligne) ==3 : # pour éviter les erreurs out of index
        if len(ligne[0]) > 0 :
           lex_general.append(ligne[0]) 
           lex_gen += " "+ligne[0]
        if len(ligne[1]) > 0 :
           lex_secu_alim.append(ligne[1]) 
        if len(ligne[2]) > 1 :
           lex_crise.append(ligne[2][:-1]) # pour enlever \n de la fin 

# Ouverture et lecture du corpus
with open(chemin_corpus, "r", encoding="utf8") as fic_corpus :
    contenu_corpus = fic_corpus.readlines()

# Liste de stopwords
stopwords_spacy = nlp.Defaults.stop_words # 507 stopwords
stopwords_sklearn = stopwords.words('french') # 157 stopwords
stopwords = []
for elt in stopwords_spacy :
    stopwords.append(elt)
for elt in stopwords_sklearn :
    if elt not in stopwords_spacy :
        stopwords.append(elt)

# Tokenisation
contenu_corpus_tokenise = []
for i in tqdm(range (0, len(contenu_corpus)), desc="Tokénisation spaCy : ") : 
    if len(contenu_corpus[i].split(";")) >=8 : # pour éviter les erreurs out of index
        article = contenu_corpus[i].split(";")[7]
        doc = nlp(article)
        text_doc = ""
        for word in doc :
            if word.text not in stopwords :
                text_doc = text_doc+" "+word.text
        contenu_corpus_tokenise.append(text_doc)

# Fonctions de vectorisation et de calcul de la similarité
def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def calcSentenceSimilarity(model,s1,s2):
    index2word_set = set(model.wv.index_to_key)
    s1_afv = avg_feature_vector(s1, model=model, num_features=300, index2word_set=index2word_set)
    s2_afv = avg_feature_vector(s2, model=model, num_features=300, index2word_set=index2word_set)
    sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    return sim

# Calcul de la similarité w2v
with open(chemin_sortie, "w", encoding="utf8") as fic_sortie :
    fic_sortie.write(contenu_corpus[0]) # ligne de titres des colonnes
    for i in tqdm(range (1, len(contenu_corpus_tokenise)), desc="Calcul de la similarité w2v : ") : 
        ligne = contenu_corpus[i].split(";")
        if len(ligne) >=8 : # pour éviter les erreurs out of index
            fic_sortie.write(ligne[0]+";") # année
            fic_sortie.write(str(round(calcSentenceSimilarity(model,lex_gen,contenu_corpus_tokenise[i]),6))+";") # calcul de sim w2v
            fic_sortie.write(ligne[2]+";") # négativité
            fic_sortie.write(ligne[3]+";") # calcul lexique SA
            fic_sortie.write(ligne[4]+";") # calcul lexique crise
            fic_sortie.write(ligne[5]+";") # régions citées
            fic_sortie.write(ligne[6]+";") # nb de mots
            fic_sortie.write(ligne[7]) # texte de l'article

### Fin du programme ###
print("\n# # # # # # # # # # # # # # # # # # # # #\n# Programme exécuté en : "+str(round((time.time()-time0)/60, 2))+" minute.s  #\n# # # # # # # # # # # # # # # # # # # # #\n")