 # Seeing Double for Clarity: Multi-View Graph Sanitization for Social Recommendation

This is the PyTorch implementation for **DuViS** proposed in the paper **Seeing Double for Clarity: Multi-View Graph Sanitization for Social Recommendation**.

##  Running environment

We develop our codes in the following environment:

- python==3.8.20
- torch==1.13.1
- numpy==1.24.3
- scikit-learn==1.3.2 
- dgl==1.0.2.cu117
- torch-sparse==0.6.17
- scipy==1.10.1

python Main.py

## Datasets



| Dataset    | # users  | # items   | # interactions |# social links  |   
|------------|----------|-----------|---------|------------|
| LastFM     | 1,892    | 17,632    | 92,834  | 25.434     | 
| Ciao       | 7,375    | 105,114   | 284,086 | 53,152     |  
| Yelp       | 16,239   | 14,284    | 169,986 | 158,590    |  
