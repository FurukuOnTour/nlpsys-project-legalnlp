# Legal NLP: Court Judgement Prediction Models

## Introduction
- Motivation: Why is this task interesting or important? 
- What other people have done? What model have previous work have used? What did they miss?  (1-2 paragraphs)
- Summarize your idea
- Summarize your results

## Our Approach/Methodology/Model 
Explain your model here how it works. 

- Input is ... 
- Output is ...
- Model description 
- Equation as necessary e.g. $\alpha_3$ 

## Dataset
- We using raw data judged and publish by The Supreme Court of Thailand.
- We labeled them up to the favorbility for the plaintiff in each case by human annotator, using Datasuar.ai as a tool for annotation.
- All data will only be labeled just one between these favorbility; Favorable: if the judgement favor plaintiff and they compensation and amount of accurseds from the start are approve and unchange by the court, Patial Favorable: if the judgement favor plaintiff but they compensation or amount of accurseds from the start are changed by the court judgement, Unfavorable: if the case has been withdrawn by the court, and Others: if the case are decide to rejudge, or the court withdrawn plaintiff accusation because of legal reasons (like lack of evidence, or claim prescibed).
- after cleaning data, our total data set is 375 tokens, 137 tokens is partially favorable, 111 tokens is unfavorable, 84 tokens is favorable, and last 43 is other. Then, we devide them into training set and develop set by 80:20 (which make tokens in traing set become 300 while dev set is 75)

| Label | Frequency |
|--------|----------|
| partially favorable | 0.365333 % |
| unfavorable | 0.296000 % |
| favorable | 0.224000 % |
| other | 0.114667 % |

## Experiment setup
- Which pre-trained model? How did you pretrain embeddings? 
- Computer. How long? 
- Hyperparameter tuning? Dropout? How many epochs? 

## Results 
How did it go?  + Interpret results. 

### Model comparison
| Model | Accuracy |
|-------|----------|
|Logistic regression | 67%| 
|**BERT** | **75%** | 


## Conclusion
- What task? What did we do? 
- Summary of results.
