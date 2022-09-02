# Extraction d'entités nommées du corpus
# Chloé Choquet
# Date de création : 05/07/2022
# Dernière MàJ : 25/07/2022

# Chemins et variables
chemin_fic_corpus = "articles_bf_calcul_w2v.csv" # corpus BF 
chemin_lexique = "localites_burkina.csv"
titre = False # True : on cherche les EN dans le titre + 1ère phrase ; False : on cherche la 1ere EN dans tout l'article
if titre : 
    nom_fic_sortie = "articles_bf_ner.csv"
else :
    nom_fic_sortie = "articles_bf_ner_all.csv"
ecrit_bassins = ["Hauts-Bassins","Hauts Bassins","Hauts-bassins","Hauts- Bassins","Haut-Bassins","Bassins","Haut-bassins","Haut Bassins"]

# Imports de librairies
import spacy
nlp = spacy.load('fr_core_news_md')
from tqdm import tqdm

# Ouverture et lecture du lexique
with open(chemin_lexique, "r", encoding="utf8") as fic_lexique :
    contenu_lexique = fic_lexique.readlines()
loc_centre = []
loc_bassins = []
loc_sahel = []
for ligne in contenu_lexique :
    ligne_split = ligne.rstrip().split(";")
    if len(ligne_split) == 3 :
        if ligne_split[0] != "" :
            loc_centre.append(ligne_split[0])
        if ligne_split[1] != "" :
            loc_bassins.append(ligne_split[1])
        loc_sahel.append(ligne_split[2])

# Ouverture et lecture du corpus
with open(chemin_fic_corpus, "r", encoding="utf8") as fic_corpus :
    contenu_corpus = fic_corpus.readlines()

# Placer les textes des articles dans une liste
txt_articles = []
regions_articles = []
for i in tqdm(range (1, len(contenu_corpus))) : # on commence à l'indice 1 pour ne pas avoir la ligne de titre ("TXT" serait sinon le 1er article)
        if len(contenu_corpus[i].split(";")) >=8 : # pour éviter les erreurs out of index
            txt_articles.append(contenu_corpus[i].split(";")[7]) # 7 : indice du texte de l'article dans le CSV
            regions_articles.append(contenu_corpus[i].split(";")[5]) # 5 : indice des régions citées dans l'article dans le CSV

# Extraire les entités nommées et les ajouter dans le corpus
with open(nom_fic_sortie, "w", encoding="utf8") as fic_sortie :
    fic_sortie.write(contenu_corpus[0]) # ligne de titre
    for i in tqdm(range(0,len(txt_articles))) :
        regions = {"Centre" : 0, "Hauts-Bassins" : 0, "Sahel" : 0}
        first_region =""
        regions_citees = ""
        if len(txt_articles[i].split(".")) >= 3 :
            if titre :
                doc = nlp(txt_articles[i].split(".")[0]+"."+txt_articles[i].split(".")[2]) # titre et première phrase de l'article (entre les deux il y a la date de publication)
            else :
                doc = nlp(txt_articles[i]) # tout l'article
            for ent in doc.ents :
                if ent.label_ == "LOC" :
                    if ent.text in loc_centre or ent.text == "Centre":
                        if titre :
                            regions["Centre"] += 1
                        else :
                            if first_region == "" :
                                first_region = "Centre"
                    elif ent.text in loc_bassins or ent.text in ecrit_bassins :
                        if titre :
                            regions["Hauts-Bassins"] += 1
                        else :
                            if first_region == "" :
                                first_region = "Hauts-Bassins"
                    elif ent.text in loc_sahel :
                        if titre :
                            regions["Sahel"] += 1
                        else :
                            if first_region == "" :
                                first_region = "Sahel"
        if titre :
            for region in regions_articles[i].split(" ") : # récupération des régions citées dans le corpus originel
                if region == "Bassins" :
                    regions["Hauts-Bassins"] += 1   
                elif region == "Centre" or region == "Sahel":
                    regions[region] += 1  
            for region in regions : # ligne avec les régions à mettre dans la colonne "REGION CITE"
                if regions[region] > 0 :
                    regions_citees += region+" "
        else :
            regions_citees = first_region
        fic_sortie.write(contenu_corpus[i+1].split(";")[0]+";"+contenu_corpus[i+1].split(";")[1]+";"+contenu_corpus[i+1].split(";")[2]+";"+contenu_corpus[i+1].split(";")[3]+";"+contenu_corpus[i+1].split(";")[4]+";"+regions_citees+";"+contenu_corpus[i+1].split(";")[6]+";"+contenu_corpus[i+1].split(";")[7])

