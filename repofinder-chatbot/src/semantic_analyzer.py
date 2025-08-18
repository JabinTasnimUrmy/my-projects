# src/semantic_analyzer.py
from sentence_transformers import SentenceTransformer, util
import torch

# Note: The first time this line runs, it will download the AI model (approx. 90MB).
# This is a one-time download.
print("Loading semantic analysis model...")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# This is our confidence threshold. If the best match is below this, we consider the idea "unique".
SIMILARITY_THRESHOLD = 0.5

def find_most_similar_project(query: str, repos: list):
    """
    Finds the most semantically similar repository description to a given query.
    """
    if not repos:
        return None

    # Create a list of project descriptions. Use the project name if the description is missing.
    corpus = [
        str(repo.get('description')) or str(repo.get('name', '')) for repo in repos
    ]

    # Generate vector embeddings for the user's query and all the repo descriptions
    query_embedding = MODEL.encode(query, convert_to_tensor=True)
    corpus_embeddings = MODEL.encode(corpus, convert_to_tensor=True)

    # Calculate the similarity score between the query and all descriptions
    cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)

    # Find the index and value of the highest score
    best_match_index = torch.argmax(cosine_scores).item()
    best_match_score = cosine_scores[0][best_match_index].item()

    print(f"(Analysis complete. Highest similarity score: {best_match_score:.2f})")

    # If the score is high enough, we have a confident match
    if best_match_score >= SIMILARITY_THRESHOLD:
        return repos[best_match_index]
    else:
        # If no project is similar enough, we return None to indicate a "unique" idea.
        return None