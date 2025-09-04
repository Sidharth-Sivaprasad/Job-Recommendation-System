import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import faiss

#Preprocess data
def preprocess_jobs(df):
    df = df.copy()
    text_cols = ['title', 'description', 'skills_desc', 'formatted_work_type', 'formatted_experience_level', 'location']
    for col in text_cols:
        df[col] = df[col].fillna('')
    df['combined_text'] = (
        df['title'] + ' ' +
        df['description'] + ' ' +
        df['skills_desc'] + ' ' +
        df['formatted_work_type'] + ' ' +
        df['formatted_experience_level'] + ' ' +
        df['location']
    )
    return df

# Embed Jobs
def embed_texts(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings, model

#Build FAISS Index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

#Get User Input
def get_user_input():
    role = input("Enter your desired role: ")
    experience = int(input("Enter your years of experience: "))
    skills = input("Enter your skills (comma-separated): ").split(',')
    location = input("Preferred job location: ")
    salary = int(input("Minimum expected salary: "))
    remote_input = input("Do you prefer remote work? (yes/no): ").strip().lower()
    remote = True if remote_input == 'yes' else False

    return {
        'role': role,
        'experience': experience,
        'skills': [skill.strip() for skill in skills],
        'location': location,
        'salary': salary,
        'remote': remote
    }

#Construct User Text
def build_user_text(user_input):
    return f"{user_input['role']} with {user_input['experience']} years experience in {' '.join(user_input['skills'])}, prefers {'Remote' if user_input['remote'] else 'In-person'} jobs in {user_input['location']} with salary above {user_input['salary']}"

#Retrieve Top-K Jobs
def retrieve_top_k(user_embedding, job_embeddings, faiss_index, k=50):
    scores, indices = faiss_index.search(user_embedding, k)
    return indices[0], scores[0]

#Rerank via PageRank
def rerank_with_pagerank(user_embedding, selected_embeddings, selected_df, top_k=10):
    all_embeddings = np.vstack([user_embedding, selected_embeddings])
    sim_matrix = cosine_similarity(all_embeddings)
    G = nx.Graph()
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            if sim_matrix[i, j] > 0.6:
                G.add_edge(i, j, weight=sim_matrix[i, j])

    if G.number_of_edges() == 0 or len(G.nodes) < 2:
        print("[Warning] Graph is disconnected. Falling back to top similarity-based results.")
        selected_df = selected_df.copy()
        selected_df['quality_score'] = [float(x) for x in range(len(selected_df), 0, -1)]
        selected_df['short_description'] = selected_df['description'].apply(lambda x: x[:120] + '...' if len(x) > 120 else x)
        return selected_df.head(top_k)[['job_id', 'title', 'company_name', 'location', 'normalized_salary', 'short_description', 'quality_score']]

    personalization = {i: 1.0 if i == 0 else 0.0 for i in G.nodes}
    try:
        pr = nx.pagerank(G, alpha=0.85, personalization=personalization)
    except ZeroDivisionError: #Fallback mechanism
        print("[Error] PageRank failed due to a disconnected graph. Falling back to similarity-based ranking.")
        selected_df = selected_df.copy()
        selected_df['quality_score'] = [float(x) for x in range(len(selected_df), 0, -1)]
        selected_df['short_description'] = selected_df['description'].apply(lambda x: x[:120] + '...' if len(x) > 120 else x)
        return selected_df.head(top_k)[['job_id', 'title', 'company_name', 'location', 'normalized_salary', 'short_description', 'quality_score']]
    sorted_nodes = sorted(((node, score) for node, score in pr.items() if node != 0), key=lambda x: x[1], reverse=True)
    top_indices = [i[0] - 1 for i in sorted_nodes[:top_k]]
    top_scores = [pr[i + 1] for i in top_indices] 
    results = selected_df.iloc[top_indices].copy()
    results['quality_score'] = top_scores
    results['short_description'] = results['description'].apply(lambda x: x[:120] + '...' if len(x) > 120 else x)
    return results[['job_id', 'title', 'company_name', 'location', 'normalized_salary', 'short_description', 'quality_score']]

#Full Recommendation Pipeline
def recommend_jobs(df, user_input, model, job_embeddings, index):
    user_text = build_user_text(user_input)
    user_embedding = model.encode([user_text], normalize_embeddings=True)
    top_indices, _ = retrieve_top_k(user_embedding, job_embeddings, index, k=50)
    top_df = df.iloc[top_indices].reset_index(drop=True)
    top_embeddings = job_embeddings[top_indices]
    top_jobs = rerank_with_pagerank(user_embedding, top_embeddings, top_df)
    return top_jobs


if __name__ == '__main__':
    df = pd.read_csv("postings.csv")
    df = preprocess_jobs(df)
    job_texts = df['combined_text'].tolist()
    job_embeddings, model = embed_texts(job_texts)
    index = build_faiss_index(job_embeddings)

    while True:
        print("\n--- New User Recommendation Session ---")
        user_input = get_user_input()
        recommendations = recommend_jobs(df, user_input, model, job_embeddings, index)
        print("\nTop Job Recommendations (with quality score):\n")
        print(recommendations.to_string(index=False))

        next_user = input("\nDo you want to enter another user? (y/n): ").strip().lower()
        if next_user != 'y':
            break

