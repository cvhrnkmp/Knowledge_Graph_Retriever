from typing import List, Dict
import re
import json
from langchain.docstore.document import Document
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, RegexpParser
import spacy
from keybert import KeyBERT
from tqdm import tqdm
# Ensure stopwords are downloaded
import nltk
from nltk.stem import WordNetLemmatizer
import spacy.tokenizer
from multimethod import multimethod



## -- Load Keyword dictionary -- ##
keyword_dictionary = pd.read_excel("/mnt/data3/christian/keywords.xlsx", index_col=0)

# Utility Functions
def find_vancouver_citation(document, metadata) -> List[any]:
    regex_pattern = r"\[\d+\]?"
    matches = re.findall(regex_pattern, document.page_content)
    refs = [match.strip("[]") for match in matches]


    source = document.metadata["source"].split("\\")[-1].split(".")[0]
    if source in metadata.keys():
        source_metadata = metadata[source]
    else:
        return {}

    references = {}
    for ref in refs:
        if ref in source_metadata["references"]:
            references[ref] = source_metadata["references"][ref]
    return references


def read_metadata(path: str):
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data


def find_technical_instructions(text: str, revision=None):
    pattern = r'\bB\d{2}-\d{3}[-/]\d{2}[-/]?[a-zA-Z\d]?\b'
    matches = re.findall(pattern, text)
    if not matches:
        return None

    results = []
    for match in matches:
        parts = re.split(r'[^-/][-/]', match)
        if len(parts) > 3:
            revision = parts[-1]
            splitted_match = re.split("[-/]+", match)
            match = "-".join(splitted_match[:-2]) + "/" + splitted_match[-2]
        results.append({"document_no": match, "revision": revision})
    return results


def find_references(document, path = "../PDF-Files/meta.json", inplace=False) -> List[any]:
    references = []
    mapped_references = []
    source = document.metadata["source"].split("/")[-1].split("\\")[-1].split(".")[0]
    metadata = read_metadata(path)
    references.extend(find_vancouver_citation(document, metadata))

    for reference in references:
        if reference in metadata[source]["references"]:
            ref_data = metadata[source]["references"][reference]
            docs = find_technical_instructions(ref_data["name"], ref_data["revision"])
            if docs:
                for doc in docs:
                    ref_data["document_no"] = doc["document_no"]
                    ref_data["revision"] = doc["revision"]
                    for key, value in metadata.items():
                        if value["no."] == ref_data["document_no"] and value["revision"] == ref_data["revision"]:
                            ref_data["document_name"] = key
                            break
                    mapped_references.append(ref_data)

    if inplace:
        document.metadata["mapped_references"] = mapped_references

    return mapped_references

def clean_and_normalize(phrase, nlp):
    phrase = re.sub(r"[^\w\s]", "", phrase)  # Remove special characters
    doc = nlp(phrase)
    
    # tokens = word_tokenize(phrase)  # Tokenize
    # lemmatized_words = [
    #     lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stopwords
    # ]
    #return " ".join(lemmatized_words)
    words = []
    for word in doc:
        if isinstance(word, str):
            result = check_keyword(word.lower())
            
            if result:
                words.append(result)
            else:
                words.append(word.lemma_.lower())
        else: 
            words.append(word.lemma_.lower())
        
    #return ' '.join([x.lemma_.lower() for x in doc]) 
    return ' '.join(words)

def check_keyword(keyword:str):
    lower_dictionary = keyword_dictionary[["org_word", "lemma"]].apply(lambda col: col.str.lower())
    search_res =  keyword_dictionary[lower_dictionary["org_word"].isin([keyword])]
    rows = len(search_res)
    if rows==1:
        normalized_keyword = search_res["lemma"].values[0]
    elif rows > 2:
        normalized_keyword = search_res["lemma"].values[0]
    else:
        search_res_lemma = keyword_dictionary[lower_dictionary["lemma"].isin([keyword])]
        rows = len(search_res_lemma)
        if rows == 1:
            normalized_keyword = search_res_lemma["lemma"].values[0]
        elif rows > 1:
            normalized_keyword = search_res_lemma["lemma"].values[0]
        else:
            normalized_keyword = None
    
    return normalized_keyword   

def check_list_of_keywords(keywords:List[str]):
    c_keywords = keywords.copy()
    permitted_keywords = []
    
    for keyword in c_keywords:
        result = check_keyword(keyword)
        if result:
            permitted_keywords.append(keyword)
            
    return permitted_keywords

def check_keywords(document:Document, inplace=False):
    print(document)
    keywords = document.metadata["keywords"]
    permitted_keywords = check_list_of_keywords(keywords)
    if inplace:
        document.metadata["keywords"] = permitted_keywords
    return permitted_keywords

# TextRank using NLTK
class TextRank:
    def __init__(self, model_name):
        # Load NLTK Stopwords for German
        self.stopwords = set(stopwords.words("german"))

        # Load a BERT model for German embeddings
        self.model_name = model_name
        if isinstance(model_name, str):
            self.model = SentenceTransformer(self.model_name)
        else:
            self.model = model_name
    
    def remove_stopwords_from_phrases(self, phrases):
        filtered_phrases = []
        lemma_phrases = []
        for phrase in phrases:
            filtered_words = [
                word for word in phrase.split() if word.lower() not in self.stopwords
            ]
            if filtered_words:
                #lemma_phrases.append([lemmatizer.lemmatize(word) for word in filtered_words])
                filtered_phrases.append(" ".join(filtered_words))
        return filtered_phrases

    def clean_and_normalize(self, phrase):
        phrase = re.sub(r"[^\w\s]", "", phrase)  # Remove special characters
        tokens = word_tokenize(phrase)  # Tokenize
        lemmatized_words = [
            word for word in tokens if word.lower() not in self.stopwords
        ]
        return " ".join(lemmatized_words)

    def extract_candidate_phrases(self, text):
        """
        Extract candidate phrases using NLTK's POS tagging and chunking.
        """
        sentences = sent_tokenize(text,language="german")  # Sentence tokenization
        candidate_phrases = []
        #grammar = "NP: {<DT>?<JJ>*<NN.*>+}"  # Noun phrase grammar
        grammar = "NP: {<NN.*>+}"  # Noun phrase grammar
        chunker = RegexpParser(grammar)

        for sentence in sentences:
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            tree = chunker.parse(pos_tags)

            # Extract noun phrases
            for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
                phrase = " ".join(word for word, tag in subtree.leaves())
                candidate_phrases.append(phrase)

        return candidate_phrases

    def text_rank_with_bert_german(self, document:Document, top_n=5, inplace = False):
        # Step 1: Preprocess and extract candidate phrases
        #print(document)
        if isinstance(document, Document):
            text = document.page_content
        else:
            text= document
        phrases = self.extract_candidate_phrases(text)
        #print(phrases)

        # Remove stopwords from phrases
        phrases = self.remove_stopwords_from_phrases(phrases)
        if not phrases:
            return []

        # Step 2: Compute BERT embeddings for each phrase
        embeddings = self.model.embed_list(phrases)

        # Step 3: Build a similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Step 4: Build a graph and apply TextRank
        #print(nx.__version__)
        graph=None
        try:
            graph = nx.from_numpy_array(similarity_matrix)
        except AttributeError:
             graph = nx.convert_matrix.from_numpy_array(similarity_matrix)
        #graph = nx.from_numpy_matrix(similarity_matrix)
        
        scores = nx.pagerank(graph, max_iter = 300)

        # Step 5: Consolidate duplicate phrases by summing their scores
        phrase_scores = {}
        for i, phrase in enumerate(phrases):
            cleaned_phrase = self.clean_and_normalize(phrase)
            if cleaned_phrase:
                if cleaned_phrase in phrase_scores:
                    phrase_scores[cleaned_phrase] += scores[i]
                else:
                    phrase_scores[cleaned_phrase] = scores[i]

        # Step 6: Sort phrases by their consolidated scores
        ranked_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 7: Extract the top N phrases
        top_phrases = [phrase for phrase, _ in ranked_phrases[:top_n]]
        
        if inplace:
            document.metadata["keywords"] = top_phrases
        
        return top_phrases

class TFKeywordExtractor():
    
    def __init__(self, language = ["german"]):
        self.language = language
        self.vectorizer = TfidfVectorizer(stop_words=self.language)
        self.stopwords = set(stopwords.words("german"))
        self.lemmatizer= WordNetLemmatizer()
      
    
    def extract_keywords_from_documents(self, documents:List[Document], inplace = True):
        """Extracts key terms using TF-IDF from a list of documents."""
        nlp = spacy.load('de_core_news_md')
        
        titles = [doc.metadata["source"] for doc in documents]
        uuids = [doc.metadata["uuid"] for doc in documents]
        texts = [clean_and_normalize(doc.page_content, nlp) for doc in documents]
        
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        keywords_per_doc = {}
        for i, title in enumerate(titles):
            tfidf_scores = tfidf_matrix[i].toarray().flatten()
            sorted_indices = tfidf_scores.argsort()[::-1]
            top_keywords = [feature_names[idx] for idx in sorted_indices[:10]]  # Top 10 keywords
            keywords_per_doc[uuids[i]] = top_keywords

        return keywords_per_doc, tfidf_matrix, uuids, self.vectorizer

    def tfidf_search(self, V, X, query, documents):
        """Performs TF-IDF based search to retrieve relevant document nodes."""
        _, tfidf_matrix, uuids, vectorizer = self.extract_keywords_from_documents(documents)
        query_vector = vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        ranked_indices = np.argsort(similarity_scores)[::-1]
        
        # Retrieve top relevant UUIDs
        top_k = 5  # Retrieve top 5 relevant passages
        top_uuids = [uuids[idx] for idx in ranked_indices[:top_k] if similarity_scores[idx] > 0.1]
        
        return top_uuids
    
class KeyBERTExtractor():
    
    def __init__(self, model = None):
        if model is None:
            self.kw_model = KeyBERT(model="Linq-AI-Research/Linq-Embed-Mistral")      
        else:
            self.kw_model = KeyBERT(model=model)
    def extract_keywords_from_documents(self, documents:List[Document], inplace = False):
        nlp = spacy.load('de_core_news_md')
        
        stopwords_list = set(stopwords.words("german"))
        for doc in tqdm(documents, desc="Extract Keyword with KeyBERT"):
            top_k = round(3/2*len(nlp.tokenizer(doc.page_content))**(1/3))
            keywords = self.kw_model.extract_keywords(clean_and_normalize(doc.page_content, nlp) , keyphrase_ngram_range=(1, 1), top_n=top_k)
            #noun_words = [nlp(word) for word in keywords]
            # noun_word = []
            # for word in keywords:
            #     tags = [token.pos_ for token in nlp(word[0])]
            #     noun_word.append([word[idx] for idx, tag in enumerate(tags) if tag not in ["VERB"]])   
            if inplace:
                doc.metadata["keywords"] = [keyword for keyword, score in keywords]
