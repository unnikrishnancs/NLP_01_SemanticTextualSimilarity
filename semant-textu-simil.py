from sentence_transformers import SentenceTransformer, util
#import numpy as np

#model=SentenceTransformer('bert-base-nli-mean-tokens')
model=SentenceTransformer('bert-base-uncased')

sent1="I like python because I can build python applications"
sent2="I like python because I can do data analytics"

#encode sentence to get their embedding
embed1=model.encode(sent1,convert_to_tensor=True)
embed2=model.encode(sent2,convert_to_tensor=True)

#compute similarity score of two embeddings
cosine_scores=util.pytorch_cos_sim(embed1,embed2)

print("Sentence1 : ",sent1)
print("Sentence2 : ",sent2)

print("Similarity Score: ",cosine_scores.item())



