Creation of a model to classify the type of jobe from a given text job description

# Goal of this project:
    * Sort emails in these classes: Job description, Alert and Others.( Asma's model)
    * Classify jobs descriptions in jobs (This model)
    * Rate cv and skills.

# We will focus on the following:
    * jobs: Data Scientist, Data Engineer,Big data developper,Data Analyst and Others(mix of other type of job)
    * datasets: Glassdoors, job_emails1 from Assan and  Kaggle 

# files' description
     * we process the data and concatenate them in the file: DataPreprocessing.ipynb
     * We did the text processing in the file: text_preprocessing_with_spacy.ipynb
     * We created and exported the pipeline( count vectorize, decision tree) in the file:
        - model_job_classification.ipynb
        - job_classification_model.py
    those two files are the same it's just the type of the file which changes
    * We load the model for deployment processin: deploy_model.py

# All dataset are in the folder data (Glasssdoors, emails from Assan, Kaggle)
# We did the text processing and the modelisation on the dataset named 'data/unbalanced.csv'.


# The file Train_rev1.csv' is larger than 1GB. If you want to use it you can download it here:
New Kaggle data: https://www.kaggle.com/code/chadalee/text-analytics-explained-job-description-data/data

# doc:
https://github.com/bellabie/spacy-tf-idf/blob/master/tf-idf.py
https://www.kaggle.com/code/sanabdriss/nlp-extract-skills-from-job-descriptions/notebook    
https://medium.com/@Olohireme/job-skills-extraction-from-data-science-job-posts-38fd58b94675   
https://www.kaggle.com/code/adarshsng/predicting-job-type-cat-using-job-description/data

To extract skills 
IDEA 1 
Edward Ross's 3 part series
 * part 1: https://skeptric.com/extract-skills-1-noun-phrase/
 * part 2: https://skeptric.com/extract-skills-2-adpositions/
 * part 3: https://skeptric.com/extract-skills-3-conjugations/