import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os
from huggingface_hub import login

huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
login(token=huggingface_token)

# Step 1: Load the datasets
def load_data():
    """Load the podcast data and ground truth data."""
    # Load podcast data from CSV instead of JSON
    try:
        podcast_data = pd.read_csv('podcast_data.csv')
        print(f"Loaded podcast data with {len(podcast_data)} episodes")
    except Exception as e:
        print(f"Error loading podcast data: {e}")
        podcast_data = pd.DataFrame()
    
    # Load ground truth data from JSON
    try:
        with open('podcast_ground_truth.json', 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        print(f"Loaded ground truth data with {len(ground_truth)} episodes")
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        ground_truth = {}
    
    return podcast_data, ground_truth

# Step 2: Initialize the embedding model
def init_model(model_name="intfloat/multilingual-e5-base"):
    """Initialize the text embedding model."""
    model = SentenceTransformer(model_name)
    
    return model

# Step 3: Create a Qdrant collection and upload podcast data
def setup_qdrant(podcast_data, model, collection_name="podcast_collection"):
    """Create a Qdrant collection and upload podcast vectors."""
    # Initialize Qdrant client
    client = QdrantClient(url="http://localhost:6333")

    # Try to create collection (ignore if exists)
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        print(f"Created new collection: {collection_name}")
    
        # Embed and upload each podcast episode
        points = []
        
        for _, row in tqdm(podcast_data.iterrows(), total=len(podcast_data), desc="Embedding podcast episodes"):
            episode_id = row['episode']
            
            # Create text from available fields
            text_parts = []
            
            # Add title
            if not pd.isna(row.get('title')):
                text_parts.append(row['title'])
            
            # Add summary if available
            if not pd.isna(row.get('summary')):
                text_parts.append(row['summary'])
                
            # Add song recommendation if available
            if not pd.isna(row.get('song_recommendation')):
                text_parts.append(f"Song: {row['song_recommendation']}")
                
            full_text = " ".join(text_parts)
            
            # Skip if no text to embed
            if not full_text:
                print(f"Skipping episode {episode_id} - No content to embed")
                continue
            
            # Embed the text
            embedding = model.encode(full_text)
            
            # Create a point
            points.append(
                models.PointStruct(
                    id=int(episode_id),
                    vector=embedding.tolist(),
                    payload={
                        'episode': int(episode_id),
                        'title': row.get('title', f"Episode {episode_id}"),
                        'summary': row.get('summary', ""),
                        'song': row.get('song_recommendation', ""),
                        'publish_date': row.get('publish_date', ""),
                        'full_text': full_text
                    }
                )
            )
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
        
        print(f"Uploaded {len(points)} episodes to Qdrant.")
        
    except Exception as e:
        print(f"Collection may already exist: {e}")
    
    return client

# Step 4: Evaluate retrieval using ground truth data
def evaluate_retrieval(client, model, ground_truth, collection_name="podcast_collection", k=5):
    """
    Evaluate retrieval performance using ground truth data.
    
    For each sentence in the ground truth data, embed it and query Qdrant.
    Check if the correct episode is in the top k results.
    """
    # Prepare to store results
    all_relevance = []
    
    # Process each episode and its data
    for episode_id, episode_data in tqdm(ground_truth.items(), desc="Evaluating episodes"):
        words = episode_data.get('word', [])
        
        for word in words:
            # Embed the sentence
            query_vector = model.encode(word)
            
            # Query Qdrant
            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=k
            )
            
            # Check if the correct episode is in the results
            result_ids = [result.id for result in search_results]
            relevance = [int(episode_id) == result_id for result_id in result_ids]
            
            all_relevance.append(relevance)
    
    return all_relevance

# Step 5: Calculate metrics
def calculate_metrics(relevance_results):
    """Calculate hit rate and MRR from relevance results."""
    # Hit rate (Recall@k)
    hit_count = sum(1 for relevance in relevance_results if True in relevance)
    hit_rate = hit_count / len(relevance_results) if relevance_results else 0
    
    # Mean Reciprocal Rank (MRR)
    mrr_sum = 0
    for relevance in relevance_results:
        for i, is_relevant in enumerate(relevance):
            if is_relevant:
                mrr_sum += 1 / (i + 1)
                break
    
    mrr = mrr_sum / len(relevance_results) if relevance_results else 0
    
    return {
        'hit_rate': hit_rate,
        'mrr': mrr,
        'total_queries': len(relevance_results),
        'successful_queries': hit_count
    }
    
# Step 6: Evaluate for different k values
def evaluate_multiple_k(client, model, ground_truth, collection_name="podcast_collection", k_values=[5, 10]):
    """Evaluate retrieval for different k values."""
    results = {}
    
    for k in k_values:
        print(f"\nEvaluating for k={k}")
        relevance_results = evaluate_retrieval(client, model, ground_truth, collection_name, k)
        metrics = calculate_metrics(relevance_results)
        
        results[k] = metrics
        print(f"Hit Rate: {metrics['hit_rate']:.4f}, MRR: {metrics['mrr']:.4f}")
    
    return results

# Main function
def main():
    # Load data
    print("Loading data...")
    podcast_data, ground_truth = load_data()

    # Initialize model
    model = init_model()

    # Setup Qdrant
    print("Setting up Qdrant...")
    client = setup_qdrant(podcast_data, model)

    # Evaluate for different k values
    print("Evaluating search performance...")
    results = evaluate_multiple_k(client, model, ground_truth)

    # Print final results
    print("\nFinal Evaluation Results:")
    print("========================")
    for k, metrics in results.items():
        print(f"k={k}:")
        print(f"  Hit Rate: {metrics['hit_rate']:.4f}")
        print(f"  MRR:      {metrics['mrr']:.4f}")
        print(f"  ({metrics['successful_queries']} out of {metrics['total_queries']} queries successful)")

    # Save results to file
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to evaluation_results.json")
    
if __name__ == "__main__":
    main()