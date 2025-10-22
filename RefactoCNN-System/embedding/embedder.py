
import numpy as np
import random

def embed_code_tokens(token_list, embed_dim=100):
    
    token_embeddings = [np.random.rand(embed_dim) for _ in token_list]
    return np.mean(token_embeddings, axis=0)  # average embedding
