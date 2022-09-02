# Topic Modeling avec LDA. 
# Création : 24/03/2022
# Dernière MàJ : 22/07/2022
# Chloé Choquet

# Imports des librairies et ressources
import datetime
import gensim
w2v_path = './Ressources_w2v/fr/fr.bin'
model = gensim.models.Word2Vec.load(w2v_path)
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
import os
from scipy import spatial
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
nlp = spacy.load('fr_core_news_md', exclude=["parser","ner","textcat"])
from statistics import mean
import time
time0 = time.time()
from tqdm import tqdm
import warnings

### PARAMETRES A MODIFIER ###

# Choix du corpus 
num_corpus = 3 # corpus par défaut : corpus BF, corpus 2 : corpus YT ; corpus 3 : corpus PADI

# Paramètres LDA 
    # nombre de topics
min_topics = 5 # nombre minimum de topics voulus
max_topics = 70 # nombre maximum de topics voulus (inclu)
pas_topics = 5 # pas entre les nombres de topics
    # fréquence minimales et maximales des termes, /!\ les listes doivent avoir la même taille
liste_minDF = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1] # Enlève les termes présents dans moins de N% d'articles
liste_maxDF = [0.9,0.89,0.88,0.87,0.86,0.85,0.84,0.83,0.82,0.81] # Enleve les termes présents dans plus de N% d'articles

tfidf = False # True : utilisation de la Term-frequence des mots pour la vectorisation des mots (TfidfVectorizer) ; False : utilisation des nombres d'occurrences des mots pour la vectorisation (CountVectorizer)

### FICHIERS ET FONCTIONS UTILISES ###

# Noms des corpus et des fichiers créés
chemin_fic_corpus = "./Corpus_TM/articles_bf.csv" # corpus BF, par défaut
nom_fic_sortie_resume = "TM_label.csv"
nom_fic_sortie_entier = "TM.csv"
nom_corpus = "articles journalistiques (articles_bf.csv)"
nom_fic_corpus = "articles_bf.csv"

if num_corpus == 2 : # corpus YT
    chemin_fic_corpus = "./Corpus_TM/corpus_YT_w2v_sup_26.csv"
    nom_fic_sortie_resume = "TM_youtube_label.csv"
    nom_fic_sortie_entier = "TM_youtube.csv"
    nom_corpus = "transcriptions de vidéos Youtube (corpus_YT_w2v_sup_26.csv)"
    nom_fic_corpus = "youtube_transcriptions.csv"
elif num_corpus == 3 : # corpus PADI
    chemin_fic_corpus = "./Corpus_TM/articles_PADI_relevant.csv"
    nom_fic_sortie_resume = "TM_PADIrel_label.csv"
    nom_fic_sortie_entier = "TM_PADIrel.csv"
    nom_corpus = "articles PADI-web (articles_PADI_relevant.csv)"
    nom_fic_corpus = "articles_PADI_relevant.csv"
nom_dossier = "Sortie_TM_"+datetime.datetime.now().strftime('%d_%m_%Hh%M_')+nom_fic_corpus[:-4]
os.mkdir(nom_dossier) # création du dossier


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

### LECTURE ET PRETRAITEMENT DU CORPUS ###

# Ouverture et lecture du corpus
with open(chemin_fic_corpus, "r", encoding="utf8") as fic_corpus :
    contenu_corpus = fic_corpus.readlines()

# Placer les textes des articles dans une liste
txt_articles = []
if num_corpus == 2 :
    for i in tqdm(range (1, len(contenu_corpus)), desc="Récupération des articles") : # on commence à l'indice 1 pour ne pas avoir la ligne de titre 
        if len(contenu_corpus[i].split(";")) >=3 : # pour éviter les erreurs out of index
            txt_articles.append(contenu_corpus[i].split(";")[3]) # 3 : indice du texte de la transcription dans le CSV
elif num_corpus == 3 :
    for i in tqdm(range (1, len(contenu_corpus)), desc="Récupération des articles") : # on commence à l'indice 1 pour ne pas avoir la ligne de titre 
        if len(contenu_corpus[i].split("\t")) >=14 : # pour éviter les erreurs out of index
            txt_articles.append(contenu_corpus[i].split("\t")[1]+contenu_corpus[i].split("\t")[2]) # 1 : titre ; 2 : texte 
else :
    for i in tqdm(range (1, len(contenu_corpus)), desc="Récupération des articles") : # on commence à l'indice 1 pour ne pas avoir la ligne de titre ("TXT" serait sinon le 1er article)
        if len(contenu_corpus[i].split(";")) >=8 : # pour éviter les erreurs out of index
            txt_articles.append(contenu_corpus[i].split(";")[7]) # 7 : indice du texte de l'article dans le CSV

# Liste de stopwords (sklearn + spaCy : 579 stopwords)
stopwords_spacy = nlp.Defaults.stop_words # 507 stopwords
stopwords_sklearn = stopwords.words('french') # 157 stopwords
stopwords_complet = []
for elt in stopwords_spacy :
    stopwords_complet.append(elt)
for elt in stopwords_sklearn :
    if elt not in stopwords_spacy :
        stopwords_complet.append(elt)
stopwords_complet = stopwords_complet + ["faire","faut","falloir","aller","marie","andré","sarah","angela","sylvie","kim","noé","antonio","jocelyne","nadine","igor","fernando","justine","marc","michel","jean","geneviève","alain","thierry","stéphane","iii","pourcent","non","www","http","com","min","euh","hum","ben","ouai","ouais","oui"] # ajout manuel de stopwords en fonction de la sortie du Topic Modeling

# Pipeline spaCy (tokénisation + stopwords + lemmatisation)
chiffres = ["0","1","2","3","4","5","6","7","8","9",",","."]
def preprocessor(text) :
    doc = nlp(text) # objet spaCy, pas une str
    liste_tokens = []
    for word in doc :
        lemma = word.lemma_  
        if lemma.lower() not in stopwords_complet and len(lemma)>2:
            lemma_is_a_number = True # par défaut on considère que le mot est un nombre
            lemma_is_an_hour = False
            for letter in lemma :
                if letter not in chiffres: # si une lettre du mot n'est pas un nombre :
                    lemma_is_a_number = False # le mot ne sera pas considéré comme un nombre
                    break
            if len(lemma)== 8 : # on cherche des heures au format "08h50min"
                if lemma[0] in chiffres and lemma[1] in chiffres and lemma[2]=="h" and lemma[3] in chiffres and lemma[4] in chiffres and lemma[5:]== "min"  :
                    lemma_is_an_hour = True
            if not lemma_is_a_number and not lemma_is_an_hour:
                liste_tokens.append(lemma)
    return (" ").join([lemma for lemma in liste_tokens])

# Prétraitement des textes
preprocess_articles = []
for i in tqdm(range(0,len(txt_articles)), desc="Prétraitement des textes : ") :
    preprocess_articles.append(preprocessor(txt_articles[i]))

### TOPIC MODELING ###

def topic_modeling(tfidf) :
    for i in tqdm(range(0,len(liste_minDF)), desc="Topic Modeling (tfidf="+str(tfidf)+") : ") :
        maxDF = liste_maxDF[i]
        minDF = liste_minDF[i]
        # Vectorisation des mots
        if tfidf :
            count_vect = TfidfVectorizer(
                max_df = maxDF,  # Enlève les termes présents dans N ou N% d'articles
                min_df = minDF,  # Enlève les termes présents dans N ou N% d'articles
                stop_words=stopwords_complet, # Liste des stopwords (sklearn + spaCy + ajouts manuels)
                lowercase=False, # Conservation des majuscules pour ensuite identifier les entités nommées
            )
            vectorisation = "TfidfVectorizer"
        else : 
            count_vect = CountVectorizer(
                max_df = maxDF,  # Enlève les termes présents dans N ou N% d'articles
                min_df = minDF,  # Enlève les termes présents dans N ou N% d'articles
                stop_words=stopwords_complet, # Liste des stopwords (sklearn + spaCy + ajouts manuels)
                lowercase=False, # Conservation des majuscules pour ensuite identifier les entités nommées
            )
            vectorisation = "CountVectorizer"
        nom_dossier_DF = nom_dossier+"\\DF"+str(i)+"_min"+str(minDF)+"_max"+str(maxDF)+"_"+vectorisation
        os.mkdir(nom_dossier_DF)
        X_train = count_vect.fit_transform(preprocess_articles)
        featureNames = count_vect.get_feature_names_out()

        # Application de la Latent Dirichlet Allocation
        liste_topics = []
        liste_nb_topics = []
        infos_resume_csv = []
        for j in range(min_topics,max_topics+pas_topics,pas_topics) :
            liste_nb_topics.append(j)
        for k in tqdm(range(0,len(liste_nb_topics)), desc ="Application de la LDA (DF"+str(i)+") : ") :
            nb_topics=liste_nb_topics[k]
            lda = LatentDirichletAllocation(
                n_components=nb_topics,     # nombre de topics
            )
            lda.fit(X_train)
            doc_topic_distrib = lda.transform(X_train) # ligne : un article ; colonne : un topic
            nb_top_words = 10
            topics_topwords = [] # liste avec tous les topwords de chaque topic (dans une liste) pour un nombre de topics donné
            liste_topwords = [] # liste de topwords pour un seul topic (permet de remplir topics_topwords, à vider à chaque boucle ci-dessous)
           
            # Ecrire les infos pour le fichier de sortie résumant les topics pour chaque itération, + score de perplexité, + score de cohérence
            info_iteration = "TOP WORDS DES "+str(nb_topics)+" TOPICS\n"
            for idx, topic in enumerate(lda.components_): # pour chaque topic
                top_words_topic = "Topic "+ str(idx)+" : "+" / ".join(featureNames[i] for i in topic.argsort()[:-nb_top_words - 1:-1]) # Top words
                for i in topic.argsort()[:-nb_top_words - 1:-1] :
                    liste_topwords.append(featureNames[i])
                topics_topwords.append(liste_topwords)
                liste_topwords = []
                info_iteration += top_words_topic+"\n"
            infos_resume_csv.append(info_iteration)
           
            # Sorties CSV avec les probas des topics pour chaque article
            with open(nom_dossier_DF+"\\"+str(nb_topics)+"topics_"+nom_fic_sortie_entier, "w", encoding="utf8") as fic_sortie_entier : # sortie csv avec les probas de chaque topic pour les articles
                fic_sortie_entier.write("Articles ("+nom_fic_corpus+")")
                for idx, topic in enumerate(lda.components_): # pour chaque topic
                    top_words_topic = " / ".join(featureNames[i] for i in topic.argsort()[:-nb_top_words - 1:-1]) # Top words
                    fic_sortie_entier.write("\tTopic "+ str(idx)+" : "+top_words_topic)
                for i in range(0, doc_topic_distrib.shape[0]) : # nombre de lignes de doc_topic_distrib = nombre d'articles
                    probas_topic = doc_topic_distrib[i]
                    fic_sortie_entier.write("\narticle "+str(i))
                    for proba in probas_topic :
                        fic_sortie_entier.write("\t"+str(round(proba, 5)))
            liste_topics.append(topics_topwords)

        # Calcul de la cohérence + graphique
        coherences = [] # cohérences pour tous les topics pour chaque itération
        for topics in liste_topics :
            coherences_intra_topics = [] # cohérences pour chaque topic au sein d'une itération
            for topic in topics : 
                moy_coherences_topwords = []
                for i in range(0,len(topic)) :
                    coherences_topwords = [] # cohérences pour chaque couple de mots au sein d'un topic
                    for j in range (i+1,len(topic)) : # calcul de sim de tous les couples de mots d'un topic
                        with warnings.catch_warnings(record=True) as w: # pour éviter une sim de 1 lorsqu'un des mots n'est pas présent dans le dico
                            sim = calcSentenceSimilarity(model,topic[i],topic[j])
                            if len(w) == 0:
                                coherences_topwords.append(sim)
                    if len(coherences_topwords) > 0 : # pour éviter une erreur avec le calcul de la moyenne si aucune sim n'a pu être calculée (mots pas dans le dico)
                        moy_coherences_topwords.append(mean(coherences_topwords))
                if len(moy_coherences_topwords) > 0 :
                    coherences_intra_topics.append(mean(moy_coherences_topwords))
            if len(coherences_intra_topics) > 0 :
                coherences.append(mean(coherences_intra_topics))
            else :
                coherences.append(0) # il faut que le nombre de scores de cohérences = nombres de topics utilisés dans les itérations
        plt.plot(liste_nb_topics,coherences, color='green', marker='o', linewidth=2, markersize=4)
        plt.xlabel("Nombre de topics")
        plt.ylabel("Cohérence w2v")
        plt.title("Scores de cohérence w2v en fonction du nombre de topics")
        plt.savefig(nom_dossier_DF+"\\coherences.png")
        plt.clf()
       
        # Fichier de sortie csv résumé
        with open(nom_dossier_DF+"\\"+nom_fic_sortie_resume, "w", encoding="utf8") as fic_sortie_resume :
            fic_sortie_resume.write("# Sortie du Topic Modeling pour le corpus "+nom_corpus+" :\n\n"+"# PARAMETRES :\n- nombres de topics : "+str(liste_nb_topics)+"\n- vectorisation : "+vectorisation+"()\n- min_df : "+str(minDF)+"\n- max_df : "+str(maxDF)+"\n\n")
            for i in range (0,len(liste_nb_topics)) :
                fic_sortie_resume.write(infos_resume_csv[i])
                fic_sortie_resume.write("COHERENCE : "+str(coherences[i])+"\n\n")
            fic_sortie_resume.write("# Programme exécuté en : "+str(round((time.time()-time0)/60, 2))+" minute.s")
    return None

# Application du topic modeling avec TfidfVectorizer et CountVectorizer
topic_modeling(tfidf)
tfidf = not tfidf
topic_modeling(tfidf)

### FIN ###
print("\n# # # # # # # # # # # # # # # # # # # # #\n# Programme exécuté en : "+str(round((time.time()-time0)/60, 2))+" minute.s  #\n# # # # # # # # # # # # # # # # # # # # #\n")
