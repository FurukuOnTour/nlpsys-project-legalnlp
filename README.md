# Legal NLP: Predicting Decisions of Tort Cases in The Supreme Court of Thailand

## Introduction
  Due to high cost of consulting legal experts and prolonged court proceedings, it is difficult for common people to access useful legal information for each of their own cases. Therefore, we have developed our court judgement prediction model in order that anyone can use this as an assistance to make decisions before starting any court proceedings. This will hopefully lower court's workload so that the court can put their effort into more complex cases.
  
  According to The 2019-2021 Statistical Report of the Court of Justice of Thailand, torts are the one among top 5 causes of action in new cases in The Supreme Court of Thailand, so in this research, we use the legal text of tort cases judged and published by The Supreme Court of Thailand.
  
  Predicting Brazilian Court Decisions (Lage-Freitas et al., 2022) used various machine learning algorithms, e.g. LSTM, GRU, BiLSTM, CNN, XGBoost, SVM, Random Forest, BERT-Imbau (BERT model pretrained with Brazilian Portuguese text), etc., to predict cases from a Brazilian court into 3 labels: “yes”, for fully favourable decisions; “partial”, for partially favourable decisions; “no”, when the appeal was denied. Due to the size of the dataset (4,043 cases), the deep learning models, i.e., LSTM, GRU, BiLSTM, and CNN, were outperformed by classic machine learning models. However, only BERT-imbau is an exception with an F1 score of 73.4%. When the labels decreased into 2 labels: "yes" and "no", XGBoost outperformed most models with an F1 score of 80.2%.
  
  Automatic Judgement Forecasting for Pending Applications of the European Court of Human Rights (Medvedeva et al., 2021) used 3 models: SVM, H-BERT, and LEGAL-BERT (BERT pretrained with various legal texts) to predict cases from the European Court of Human Rights into 2 labels: "violation" and "non-violation". Due to the length of the case text, SVM performed better than H-BERT and LEGAL-BERT with an F1 score around 61-65%.
  
  In this project, we use 375 samples of legal text of tort cases judged and published by The Supreme Court of Thailand. Those cases are labeled into 4 labels: "favorable", "partially favorable", "unfavorable", and "other" by us. Those samples are used in 3 tasks: verdict identification, verdict classification, verdict prediction using case's plains. The models used in the verdict identification task is CRF. The models used in the verdict classification and the verdict prediction tasks are Logistic Regression, CNN, and WangchanBERTa (BERT pretrained with Thai text). The result is that in the verdict prediction task, WangchanBERTa outperformed other models, while in the verdict classification task, Logistic Regression was slightly better than WangchanBERTa and much better than CNN.

## Our Approach
In this project, we have divided our task into 3 tasks: verdict identification, verdict classification, verdict prediction.

1. Verdict identification: This model takes a whole text of a tort case as an input. It will identify which part of the text is the verdict of the case and return in IOB format. We used CRF to achieve this task.
2. Verdict classification: This model takes a verdict part of a tort case as an input. It will classify the verdict into 4 labels: "favorable", "partially favorable", "unfavorable", and "other". We experimented this task with a few types of machine learning algorithms: Logistic Regression, Convolutional Neural Network (CNN) and WangchanBERTa.
3. Verdict prediction: This model takes a case's plain part of a tort case as an input. It will classify the verdict into 4 labels: "favorable", "partially favorable", "unfavorable", and "other". We also experimented this task with Logistic Regression, Convolutional Neural Network (CNN) and WangchanBERTa.

## Dataset
We used raw data judged and published by The Supreme Court of Thailand. We labeled them up to the favorbility for the plaintiff in each case by human annotator using Datasuar.ai as a tool for annotation. All data will only be labeled just one between these favorbility; Favorable: if the judgement favor plaintiff and their compensation and amount of defandants from the start are approved and unchanged by the court, Patially Favorable: if the judgement favor plaintiff but their compensation or amount of defandants from the start are changed by the court judgement, Unfavorable: if the case has been withdrawn by the court, and Others: if the case are decided to rejudge, or the court have withdrawn plaintiff accusation because of legal reasons (like lack of evidence, or claim prescibed). After cleaning data, our total data set is 375 cases, 137 cases is partially favorable, 111 cases is unfavorable, 84 cases is favorable, and last 43 is other. Then, we devided them into training set and development set by 80:20, which makes the data in training set become 300 while the development set is 75.

| Label | Frequency |
|--------|----------|
| partially favorable | 36.5333 % |
| unfavorable | 29.6000 % |
| favorable | 22.4000 % |
| other | 11.4667 % |

## Experiment setup
First, we start train Conditonal Random Field (CRF) model using modify train and dev set (we change label to plain and verdict instead of favorbility) that we prepare for training and evaluation. We done this step to create a model that could help us automatically labeled where the verdict is in raw data.

Second, we train and compete the result to find the best classifier model that could predict favorblility with most accuracy (F1) using only verdict as an input. In this process, we intend to train a model that could annotate favorbility instead of human annonator. We try 3 different model for our candidate; Logistic Regression, Convolutional Neural Network (CNN), and BERT (we use WangchanBERTa as a base for this task).
    
Third, we train and compete the result to find the best classifier model that could predict favorblility with most accuracy (F1) using only case's plain as an input. We also the same try 3 different model for our candidate with unchange parameter tuning; Logistic Regression, Convolutional Neural Network (CNN) with Thai National Corpus word embedding, and BERT (we use WangchanBERTa as a base for this task). 

Both task use the same parameter and word embedding in the table under this paragragh. Both CNN and BERT use around 5-7 minute for training, while Logistic Regression use around 2-4 minutes.

| CNN | WangchanBERTa |
|----------|---------------|
| filters = 150 | learning_rate=3e-5 |
| kernel_size = 45 | per_device_train_batch_size=96 |
| hidden_dims = 150 | per_device_eval_batch_size=96 |
| 100-unit TNC word embedding | num_train_epochs=90 |
| max word len = 600 | drop_out = 0.4 |

But after the testing, the result from verdict-input classifier models are surprisingly not impressive for us. We presume that it's because verdict usually state only the court stand they judgement, or draw back to use old verdict from trail court or appellate court without any further information. To understand what the supreme court really judge, we need to look back at appellate court judgement or even untill the first judgement from trail court. To solve this problem, we train and compete another classifier model that could predict favorblility using full court record (contain both plain and verdict) as an input with the same candidate model and parameter tuning. We presume that by using full court record, the models should understand more context of the case and can predict more precisely.

## Results 
Our CRF model for sequence tagging are very successful with F1 score at 0.96 as well as WangchanBERTa that best other models F1 score at 0.83 in plain-input classifier model. But in court-record-input classifier, it turns out that Logistic Regression is the champion with 0.76 of F1 score, better than WangchanBERTa F1 score at 0.74 or CNN at 0.66

### Model comparison for plain-input classifier model
| Model | Accuracy (F1) |
|-------|----------|
|Logistic regression | 78%|
|CNN| 80% |
|**WangchanBERTa** | **83%** |

### Model comparison for court-record-input classifier model
| Model | Accuracy (F1) |
|-------|----------|
|**Logistic regression** | **76%**|
|CNN| 66% |
|WangchanBERTa | 74% |

Even though WangchanBERTa can do quite good on both task. it's still not perform at best as it could as there's a maximum input length in WangchanBERTa model, which is 256 tokens. We could change maximum input lenght to much more bigger number like 512 or 1024 as long as we don't have to manage CUDA memory in our Google Colab Session.

## Conclusion
After using 375 samples of legal text of tort cases to do verdict identification, verdict classification, and verdict prediction tasks with various models, we have found that our CRF model for verdict identification was successful with an F1 score of 96%, and WangchanBERTa performed very well in verdict perdiction task with an F1 score of 83%. However, in the verdict classification task, Logistic Regression was slightly better than WangchanBERTa and much better than CNN with an F1 score of 76%, which is still not quite impressive for us. We need to study further for verdict classification task in order that we can automate the data labeling process.
