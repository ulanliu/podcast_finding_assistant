import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import os
import sys
from typing import List, Dict, Tuple, Any, Optional

class SemanticSearchRetriever:
    """
    Two-stage semantic search system with separate models:
    1. Initial vector search using intfloat/multilingual-e5-base for candidate retrieval
    2. Re-ranking using BAAI/bge-reranker-base for more precise semantic matching
    """
    def __init__(self, 
                podcast_data_path='podcast_data.csv',
                embedding_model_name="intfloat/multilingual-e5-base",
                reranker_model_name="BAAI/bge-reranker-base",
                qdrant_url="http://localhost:6333",
                collection_name="podcast_collection"):
        
        # Load podcast metadata
        self.collection_name = collection_name
        self.podcast_data = self.load_podcast_data(podcast_data_path)
        self.embedding_cache = {}  # Cache for query embeddings
        self.episode_text_cache = {}  # Cache for episode texts
        self.vector_search_cache = {}  # Cache for vector search results
        
        # Initialize embedding model for vector search
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            print(f"Successfully loaded embedding model: {embedding_model_name}")
        except Exception as e:
            print(f"Error loading primary embedding model: {e}")
            fallback_models = [
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "distiluse-base-multilingual-cased-v1"
            ]
            for fallback_model in fallback_models:
                try:
                    print(f"Trying fallback embedding model: {fallback_model}")
                    self.embedding_model = SentenceTransformer(fallback_model)
                    print(f"Successfully loaded fallback embedding model: {fallback_model}")
                    break
                except Exception as e2:
                    print(f"Error loading fallback embedding model {fallback_model}: {e2}")
            else:
                raise ValueError("Failed to load any embedding model.")
        
        # Initialize reranker model
        try:
            self.reranker_model = CrossEncoder(reranker_model_name)
            print(f"Successfully loaded reranker model: {reranker_model_name}")
        except Exception as e:
            print(f"Error loading primary reranker model: {e}")
            fallback_rerankers = [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "cross-encoder/stsb-roberta-base"
            ]
            for fallback_model in fallback_rerankers:
                try:
                    print(f"Trying fallback reranker model: {fallback_model}")
                    self.reranker_model = CrossEncoder(fallback_model)
                    print(f"Successfully loaded fallback reranker model: {fallback_model}")
                    break
                except Exception as e2:
                    print(f"Error loading fallback reranker model {fallback_model}: {e2}")
            else:
                raise ValueError("Failed to load any reranker model.")
        
        # Connect to Qdrant
        try:
            self.client = QdrantClient(url=qdrant_url)
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                raise ValueError(f"Collection {collection_name} not found in Qdrant. Please ensure it exists.")
            
            print(f"Successfully connected to Qdrant collection: {collection_name}")
        except Exception as e:
            raise ConnectionError(f"Error connecting to Qdrant: {e}")
        
    def load_podcast_data(self, path):
        """Load podcast metadata"""
        try:
            df = pd.read_csv(path)
            print(f"Loaded podcast data with {len(df)} episodes")
            return df
        except Exception as e:
            print(f"Error loading podcast data: {e}")
            return pd.DataFrame()
            
    def vector_search(self, query, top_k=20):
        """Initial candidate retrieval using vector search in Qdrant with embedding model"""
        cache_key = (query, top_k)
        if cache_key in self.vector_search_cache:
            return self.vector_search_cache[cache_key]
        
        # Get embedding (with caching)
        if query in self.embedding_cache:
            query_vector = self.embedding_cache[query]
        else:
            query_vector = self.embedding_model.encode(query)
            self.embedding_cache[query] = query_vector
        
        # Use Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        
        # Format results as expected
        results = [(res.id, res.score) for res in search_results]
        
        # Cache result before returning
        self.vector_search_cache[cache_key] = results
        return results
            
    def get_episode_text(self, episode_id):
        """Get text representation for an episode from Qdrant or podcast_data"""
        # First try to get from Qdrant
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[episode_id]
            )
            
            if points:
                # Try to get full_text if available
                if points[0].payload.get('full_text'):
                    return points[0].payload.get('full_text')
                
                # Otherwise reconstruct from topics and song
                topics = points[0].payload.get('topics', [])
                song = points[0].payload.get('song', "")
                
                topics_text = '; '.join(topics) if topics else ""
                full_text = topics_text
                
                if song and song != "null" and not pd.isna(song):
                    full_text += f"; Song: {song}"
                    
                return full_text
        except Exception as e:
            print(f"Error retrieving episode {episode_id} from Qdrant: {e}")
        
        # Fallback to podcast_data.csv
        episode_rows = self.podcast_data[self.podcast_data['episode'] == episode_id]
        if not episode_rows.empty:
            episode_row = episode_rows.iloc[0]
            
            # Construct a text representation from available fields
            text_parts = []
            
            # Add title
            if not pd.isna(episode_row.get('title')):
                text_parts.append(episode_row['title'])
            
            # Add summary if available
            if not pd.isna(episode_row.get('summary')):
                text_parts.append(episode_row['summary'])
                
            # Add song recommendation if available
            if not pd.isna(episode_row.get('song_recommendation')):
                text_parts.append(f"Song: {episode_row['song_recommendation']}")
                
            return " ".join(text_parts)
            
        return f"Episode {episode_id}"
            
    def get_episode_title(self, episode_id):
        """Get episode title if available"""
        episode_rows = self.podcast_data[self.podcast_data['episode'] == episode_id]
        if not episode_rows.empty:
            return episode_rows.iloc[0].get('title', f"Episode {episode_id}")
        return f"Episode {episode_id}"
    
    def get_episode_details(self, episode_id):
        """Get detailed information about an episode"""
        episode_rows = self.podcast_data[self.podcast_data['episode'] == episode_id]
        if not episode_rows.empty:
            row = episode_rows.iloc[0]
            return {
                'episode': episode_id,
                'title': row.get('title', f"Episode {episode_id}"),
                'summary': row.get('summary', ""),
                'song_recommendation': row.get('song_recommendation', ""),
                'publish_date': row.get('publish_date', "")
            }
        return {'episode': episode_id, 'title': f"Episode {episode_id}"}
    
    def rerank_with_cross_encoder(self, query, candidates, top_k=5):
        """Re-rank candidates using the cross-encoder reranker model"""
        # Collect all candidates at once
        candidate_texts = []
        candidate_ids = []
        
        for episode_id, _ in candidates:
            if episode_id in self.episode_text_cache:  # See caching optimization below
                episode_text = self.episode_text_cache[episode_id]
            else:
                episode_text = self.get_episode_text(episode_id)
                self.episode_text_cache[episode_id] = episode_text
                
            if not episode_text:
                continue
                
            candidate_texts.append(episode_text)
            candidate_ids.append(episode_id)
        
        # Process all pairs in a single batch (instead of one by one)
        input_pairs = [[query, text] for text in candidate_texts]
        scores = self.reranker_model.predict(input_pairs, batch_size=32)  # Add batch_size parameter
        
        
        # Prepare candidate pairs for reranking
        for episode_id, _ in candidates:
            episode_text = self.get_episode_text(episode_id)
            if not episode_text:
                continue
                
            candidate_texts.append(episode_text)
            candidate_ids.append(episode_id)
            
        # No valid candidates
        if not candidate_texts:
            return []
            
        # Prepare input pairs for cross-encoder
        input_pairs = [[query, text] for text in candidate_texts]
        
        # Get reranker scores
        try:
            scores = self.reranker_model.predict(input_pairs)
            
            # Combine scores and IDs
            ranked_results = list(zip(candidate_ids, scores))
            
            # Sort by score (descending)
            return sorted(ranked_results, key=lambda x: x[1], reverse=True)[:top_k]
        except Exception as e:
            print(f"Error in reranking: {e}")
            # Fallback to original order if reranking fails
            return [(id, 0.0) for id in candidate_ids][:top_k]
    
    def two_stage_semantic_search(self, query, first_stage_k=20, top_k=5):
        """
        Perform two-stage semantic search:
        1. Get initial candidates using vector search with embedding model
        2. Re-rank candidates using cross-encoder reranker model
        
        Args:
            query: Search query
            first_stage_k: Number of candidates to retrieve in first stage
            top_k: Number of final results to return
            
        Returns:
            List of (episode_id, score) tuples
        """
        # First stage: Get candidates using vector search
        candidates = self.vector_search(query, top_k=first_stage_k)
        
        # Second stage: Re-rank candidates using cross-encoder
        return self.rerank_with_cross_encoder(query, candidates, top_k=top_k)
    
    def evaluate_semantic_search(self, 
                                ground_truth_path='podcast_ground_truth.json',
                                k_values=[5, 10]):
        """
        Evaluate semantic search performance using ground truth data
        
        Args:
            ground_truth_path: Path to ground truth data
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        # Load ground truth data
        try:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
        except Exception as e:
            print(f"Error loading ground truth data: {e}")
            return {}
            
        # Create test queries from ground truth
        test_queries = []
        for episode_id, episode_data in ground_truth.items():
            words = episode_data.get('word', [])
            for word in words:
                test_queries.append((word, int(episode_id)))
                
        print(f"Created {len(test_queries)} test queries from ground truth")
        
        # Initialize results
        results = {
            "vector_search": {},  # Initial vector search
            "cross_encoder_rerank": {}  # Two-stage search with cross-encoder
        }
            
        # Evaluate for different k values
        for k in k_values:
            print(f"\nEvaluating for k={k}")
            
            # Evaluate vector search
            print("Evaluating vector search...")
            vector_relevance = []
            for query, true_id in tqdm(test_queries, desc="Vector search"):
                results_ids = [id for id, _ in self.vector_search(query, top_k=k)]
                relevance = [true_id == id for id in results_ids]
                vector_relevance.append(relevance)
                
            vector_metrics = calculate_metrics(vector_relevance)
            results["vector_search"][k] = vector_metrics
            print(f"Vector Search: Hit Rate: {vector_metrics['hit_rate']:.4f}, MRR: {vector_metrics['mrr']:.4f}")
            
            # Evaluate two-stage search with cross-encoder
            print("Evaluating cross-encoder reranking...")
            rerank_relevance = []
            
            for query, true_id in tqdm(test_queries, desc="Cross-encoder rerank"):
                search_results = self.two_stage_semantic_search(
                    query, first_stage_k=20, top_k=k
                )
                results_ids = [id for id, _ in search_results]
                relevance = [true_id == id for id in results_ids]
                rerank_relevance.append(relevance)
                
            rerank_metrics = calculate_metrics(rerank_relevance)
            results["cross_encoder_rerank"][k] = rerank_metrics
            print(f"Cross-Encoder Rerank: Hit Rate: {rerank_metrics['hit_rate']:.4f}, MRR: {rerank_metrics['mrr']:.4f}")
                
        return results
        
    def search_and_explain(self, query, top_k=5, first_stage_k=20, verbose=True):
        """
        Perform semantic search and explain results
        
        Args:
            query: Search query
            top_k: Number of results to return
            first_stage_k: Number of candidates to retrieve in first stage
            verbose: Whether to print detailed information
            
        Returns:
            List of search results with explanations
        """
        # First stage: Get candidates using vector search
        candidates = self.vector_search(query, top_k=first_stage_k)
        
        if verbose:
            print(f"\nQuery: '{query}'")
            print("-" * 80)
            print(f"First stage: Retrieved {len(candidates)} candidates using vector search")
            
        # Second stage: Re-rank candidates
        final_results = self.two_stage_semantic_search(
            query, first_stage_k=first_stage_k, top_k=top_k
        )
        
        # Prepare detailed results with explanations
        detailed_results = []
        
        for rank, (episode_id, score) in enumerate(final_results):
            # Get episode text and details
            episode_text = self.get_episode_text(episode_id)
            episode_details = self.get_episode_details(episode_id)
            
            # Get vector search rank
            vector_rank = next(
                (idx+1 for idx, (id, _) in enumerate(candidates) if id == episode_id), 
                "Not in top candidates"
            )
            
            # Find improvement in rank
            rank_improvement = None
            if isinstance(vector_rank, int):
                rank_improvement = vector_rank - (rank + 1)
                
            # Prepare explanation
            explanation = {
                "rank": rank + 1,
                "episode_id": episode_id,
                "title": episode_details['title'],
                "score": score,
                "vector_search_rank": vector_rank,
                "rank_improvement": rank_improvement,
                "text_preview": episode_text[:200] + "..." if len(episode_text) > 200 else episode_text,
                "summary": episode_details.get('summary', ''),
                "song_recommendation": episode_details.get('song_recommendation', ''),
                "publish_date": episode_details.get('publish_date', '')
            }
            
            detailed_results.append(explanation)
            
            # Print explanation if verbose
            if verbose:
                print(f"\nRank {rank+1}: Episode {episode_id} - {episode_details['title']}")
                print(f"  Cross-encoder score: {score:.4f}")
                print(f"  Initial vector search rank: {vector_rank}")
                if rank_improvement is not None and rank_improvement > 0:
                    print(f"  ↑ Improved by {rank_improvement} positions after reranking")
                print(f"  Text preview: {explanation['text_preview']}")
                if explanation['summary']:
                    print(f"  Summary: {explanation['summary'][:100]}..." if len(explanation['summary']) > 100 else explanation['summary'])
                if explanation['song_recommendation'] and str(explanation['song_recommendation']) != 'nan':
                    print(f"  Song recommendation: {explanation['song_recommendation']}")
                
        return detailed_results

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

def setup_qdrant(df, model, collection_name="podcast_collection", qdrant_url="http://localhost:6333"):
    """
    Create and populate a Qdrant collection using podcast data from CSV.
    
    Args:
        df: Pandas DataFrame with podcast data
        model: SentenceTransformer model for embeddings
        collection_name: Name for the Qdrant collection
        qdrant_url: URL of the Qdrant server
        
    Returns:
        QdrantClient instance
    """
    # Initialize Qdrant client
    client = QdrantClient(url=qdrant_url)
    
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
    except Exception as e:
        print(f"Collection may already exist: {e}")
    
    # Embed and upload each podcast episode
    points = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding podcast episodes"):
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
    return client

def interactive_search():
    """Run interactive search"""
    retriever = SemanticSearchRetriever()
    
    print("\nReady to search! Enter queries below, or type 'exit' to quit.")
    
    # Interactive search
    while True:
        query = input("\nEnter search query (or 'exit' to quit): ")
        if query.lower() in ('exit', 'quit', 'q'):
            break
            
        try:
            retriever.search_and_explain(query, top_k=5)
        except Exception as e:
            print(f"Error during search: {e}")

def evaluate_search():
    """Run evaluation"""
    retriever = SemanticSearchRetriever()
    
    print("\nRunning evaluation...")
    results = retriever.evaluate_semantic_search(
        k_values=[5, 10]
    )
    
    # Print final results
    print("\nFinal Evaluation Results:")
    print("========================")
    for method, method_results in results.items():
        print(f"\n{method.upper()}:")
        for k, metrics in method_results.items():
            print(f"  k={k}:")
            print(f"    Hit Rate: {metrics['hit_rate']:.4f}")
            print(f"    MRR:      {metrics['mrr']:.4f}")
            print(f"    ({metrics['successful_queries']} out of {metrics['total_queries']} queries successful)")
    
    # Save results to file
    with open('semantic_search_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to semantic_search_evaluation.json")

def initialize_system():
    """Initialize the search system by setting up Qdrant with podcast data"""
    # Load the podcast data
    try:
        df = pd.read_csv('podcast_data.csv')
        print(f"Loaded podcast data with {len(df)} episodes")
    except Exception as e:
        print(f"Error loading podcast data: {e}")
        return
    
    # Initialize the embedding model
    try:
        model = SentenceTransformer("intfloat/multilingual-e5-base")
        print("Successfully loaded embedding model")
    except Exception as e:
        print(f"Error loading primary model: {e}")
        try:
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            print("Successfully loaded fallback model")
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
            return
    
    # Setup Qdrant collection
    setup_qdrant(df, model)
    
    print("\nSystem initialization complete. You can now run search or evaluation.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic Search with Cross-Encoder Reranking")
    parser.add_argument("--mode", choices=["search", "evaluate", "init"], default="search",
                        help="Run mode: search, evaluate, or init")
    args = parser.parse_args()
    
    if args.mode == "init":
        initialize_system()
    elif args.mode == "search":
        interactive_search()
    else:
        evaluate_search()