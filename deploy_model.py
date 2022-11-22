import pandas as pd
import spacy
import string
import joblib


# load a new spacy model
nlp = spacy.load("en_core_web_lg")

# add somes stopwords in the default list of spacy
nlp.Defaults.stop_words |= {"or","per","like",'-','_','',
                            '–','[]','\n','\n\n','\n\n ','i.e.'}

# Create our list of stopwords
stopWords= spacy.lang.en.stop_words.STOP_WORDS #set(stopwords.words('english'))


def text_preprocess(text):
    nlp.max_length = 2030000 # To raise the max legnth of word
    
### Creation of function that we'll use to process data
# Lemmatization
    def lemmatize_word(text):
        """ lemmatise words to give his root for example did becomes do
        input: text that contains Tokens
        output: A list of lemmatized tokens
        """
        lemma_word = [] 
        for token in text:
            lemma_word.append(token.lemma_)
        return lemma_word
    
    
# Split words that ontains character and correct them 
    def check_character_in_words(text):
        """ Split words that ontains character and correct them 
        input: A list of tokens with characters
        output: A list of tokens splitted on the character
        """
        charact = ["\n", ":", '$']
        for words in text:
            for chars in charact:
                if chars in words:
                    text.remove(words)
                    words = words.replace(chars," ")
                    words = words.split()
                    text.extend(words)

        return text 
    
# Remove punctuation    
    def remove_punct(text):
        """
        Remove punctuation from text (List of tokens)
        input: A list of tokens with punctuation
        output: A list of tokens without punctuation
        """
        l=[]
        for word in text:
            if not word in string.punctuation:# list of punctuation
                l.append(word)
        # resultat=" ".join(l)   
        return l


# Remove stopwords
    def remove_stopword(liste,stopWords):
        """
        Remove stopwords from a list of tokens
        input:A list of tokens with stopwords
        output:A list of tokens without stopwords
        """
        list_tokens = [tok.lower() for tok in liste]
        l=[]
        for word in list_tokens:  
            if word not in stopWords:
                l.append(word)
        return l
    
# Remove duplicated wods 
    # In the text the world sometimes repeated twice or more. 
    # For example slary in the title of the job and the description
    
    def remove_duplicates(text):
        
        """Remove duplicated words in each elements of the list
        input: list
        output: list
        """
        l=[]
        [l.append(x) for x in text if x not in l]
        resultat=" ".join(l)
        return resultat

    
### Process the data

    # Tokenization
    doc = nlp(text)
    lemmatize_text = lemmatize_word(doc)
    checked_text = check_character_in_words(lemmatize_text)
    removed_punctuation_text = remove_punct(checked_text)
    removed_stopwords_punctuation = remove_stopword(removed_punctuation_text,
                                                    stopWords)
# Remove duplicated wods 
    clean_text = remove_duplicates(removed_stopwords_punctuation)
#     resultat=" ".join(clean_text)
    return(clean_text) 
    
    

def model(email):
    """
    tokenize the text, load the model and predict the job title
    input: string
    output: job_title
    """
    processed_email = text_preprocess(email)
    model = joblib.load("model.joblib")
    job_title = model.predict([processed_email])
    return(job_title)



