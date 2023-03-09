# Sentence Transformer transfer learning and downstream task evaluation for multilingual model
*NLP_Unibo_Project_Work*

### What is it?

In this repo you will find our Project Work for UniBo NLP course 2021/2022. In particular we used transfer learning (teacher-student architecture) to distill knowledge from a Sentence Transformer trained on English textual data, we obtained this multilingual student model:
* A [multilingual sentence transformer](https://huggingface.co/airnicco8/xlm-roberta-en-it-de)

This link will redirect to the model card on the Hugging Face hub, there will be instructions to deploy the model with 2 lines of code!

### Transfer Learning

For the knowledge distillation process we referred to [this](https://towardsdatascience.com/a-complete-guide-to-transfer-learning-from-english-to-other-languages-using-sentence-embeddings-8c427f8804a9) blog post. We used paraphrase-distilroberta-base-v2 as teacher and a XLM-RoBERTa-base model as student. The process led to a multilingual sentence transformer which is able to encode sentences in English, Italian and German.
 
### Downstream Tasks

After the transfer learning phase we needed suitable tasks to evaluate the performance and/or adaptability of our student model. In particular we chose:
* **Text Similarity:** this is the most straight forward task, since the model already give as output an embedding of a sentence and we can easily calculate any kind of distances between a number of sentence embeddings to estimate similarity. This kind of evaluation is done directly in the `Project_work.ipynb` notebook.
Data sources divided for language:
    * <ins>ENG</ins>, <ins>ITA</ins>, <ins>GER</ins>: [STSb](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)
* **Text Classification:** The sentence embeddings produced by our student model can be used to classify pieces of text. In particular we fine-tune and evaluate our model on differente tasks for each language. This evaluation is done in the `SetFit_Classification.ipynb` notebook. The process is taken from this [post](https://towardsdatascience.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e).
Data sources divided for language:
    * <ins>ENG</ins>: [SST2](https://github.com/clairett/pytorch-sentiment-classification)
    * <ins>ITA</ins>: [Sentipolc2016](http://www.di.unito.it/~tutreeb/sentipolc-evalita16/index.html)
    * <ins>GER</ins>: [GNAD10k](https://github.com/goerlitz/nlp-classification/tree/main/notebooks/10kGNAD)
