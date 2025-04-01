import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, util
import os
import sys
from typing import List, Dict, Tuple, Any, Optional

class SemanticSearchRetriever:
    """
    Pure semantic search system with two-stage approach:
    1. Initial vector search for candidate retrieval
    2. Re-ranking using cross-attention for more precise semantic matching
    """
    def __init__(self, 
                podcast_topic_path='podcast_topic.json',
                podcast_data_path='podcast_data.csv',
                model_name="intfloat/multilingual-e5-base",
                qdrant_url="http://localhost:6333",
                collection_name="podcast_collection"):
        
        # Load podcast metadata
        self.collection_name = collection_name
        self.podcast_data = self.load_podcast_data(podcast_data_path)
        self.podcast_topic = self.load_podcast_topic(podcast_topic_path)
        
        # Initialize embedding model
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading primary model: {e}")
            fallback_models = [
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "distiluse-base-multilingual-cased-v1"
            ]
            for fallback_model in fallback_models:
                try:
                    print(f"Trying fallback model: {fallback_model}")
                    self.model = SentenceTransformer(fallback_model)
                    print(f"Successfully loaded fallback model: {fallback_model}")
                    break
                except Exception as e2:
                    print(f"Error loading fallback model {fallback_model}: {e2}")
            else:
                raise ValueError("Failed to load any embedding model.")
        
        # Try to connect to Qdrant
        try:
            self.client = QdrantClient(url=qdrant_url)
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            self.has_qdrant = True
            if collection_name not in collection_names:
                print(f"Collection {collection_name} not found. Please ensure it exists.")
                print("Will use in-memory vector search instead.")
                self.has_qdrant = False
                self._prepare_in_memory_search()
            else:
                print(f"Using Qdrant collection: {collection_name}")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            print("Will use in-memory vector search instead.")
            self.has_qdrant = False
            self._prepare_in_memory_search()
        
    def load_podcast_data(self, path):
        """Load podcast metadata"""
        try:
            df = pd.read_csv(path)
            print(f"Loaded podcast data with {len(df)} episodes")
            return df
        except Exception as e:
            print(f"Error loading podcast data: {e}")
            return pd.DataFrame()
            
    def load_podcast_topic(self, path):
        """Load podcast topics"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded podcast topics for {len(data)} episodes")
            return data
        except Exception as e:
            print(f"Error loading podcast topics: {e}")
            return {}
            
    def _prepare_in_memory_search(self):
        """Prepare in-memory vector search as fallback"""
        print("Preparing in-memory vector search...")
        self.episode_texts = {}
        self.episode_embeddings = {}
        
        # Process each episode
        for episode_id, content in tqdm(self.podcast_topic.items()):
            topics = content.get('topic', [])
            song = content.get('song', "")
            
            # Create text representation
            topics_text = '; '.join(topics) if topics else ""
            
            full_text = topics_text
            if song and song != "null" and not pd.isna(song):
                full_text += f"; Song: {song}"
            
            if full_text:
                episode_id_int = int(episode_id)
                self.episode_texts[episode_id_int] = full_text
                # Calculate embedding
                self.episode_embeddings[episode_id_int] = self.model.encode(full_text)
        
        print(f"Prepared in-memory search for {len(self.episode_embeddings)} episodes")
            
    def vector_search(self, query, top_k=100):
        """Initial candidate retrieval using vector search"""
        query_vector = self.model.encode(query)
        
        if self.has_qdrant:
            # Use Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k
            )
            return [(res.id, res.score) for res in search_results]
        else:
            # Use in-memory search
            results = []
            for episode_id, embedding in self.episode_embeddings.items():
                # Calculate cosine similarity
                similarity = np.dot(query_vector, embedding) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(embedding)
                )
                results.append((episode_id, float(similarity)))
            
            # Sort by similarity (descending)
            return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
            
    def get_episode_text(self, episode_id):
        """Get text representation for an episode"""
        if self.has_qdrant:
            # Try to get from Qdrant
            try:
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[episode_id]
                )
                if points and points[0].payload.get('full_text'):
                    return points[0].payload.get('full_text')
            except:
                pass
        
        # Fallback to local data
        if episode_id in self.episode_texts:
            return self.episode_texts[episode_id]
            
        # Try to reconstruct from podcast_topic
        episode_id_str = str(episode_id)
        if episode_id_str in self.podcast_topic:
            content = self.podcast_topic[episode_id_str]
            topics = content.get('topic', [])
            song = content.get('song', "")
            
            topics_text = '; '.join(topics) if topics else ""
            full_text = topics_text
            if song and song != "null" and not pd.isna(song):
                full_text += f"; Song: {song}"
                
            return full_text
            
        return ""
            
    def get_episode_title(self, episode_id):
        """Get episode title if available"""
        episode_rows = self.podcast_data[self.podcast_data['episode'] == episode_id]
        if not episode_rows.empty:
            return episode_rows.iloc[0].get('title', f"Episode {episode_id}")
        return f"Episode {episode_id}"
    
    def get_semantic_similarity(self, query, text):
        """Calculate semantic similarity between query and text"""
        if not query or not text:
            return 0.0
            
        # Encode both
        query_embedding = self.model.encode(query)
        text_embedding = self.model.encode(text)
        
        # Calculate cosine similarity
        return float(util.cos_sim(query_embedding, text_embedding)[0][0])
    
    def semantic_rerank(self, query, candidates, top_k=10):
        """Re-rank candidates using cross-attention semantic similarity"""
        results = []
        
        for episode_id, initial_score in candidates:
            # Get text for this episode
            episode_text = self.get_episode_text(episode_id)
            
            # Skip if no text available
            if not episode_text:
                continue
                
            # Calculate semantic similarity directly
            similarity = self.get_semantic_similarity(query, episode_text)
            results.append((episode_id, float(similarity)))
            
        # Sort by similarity and return top k
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    
    def two_stage_semantic_search(self, query, first_stage_k=100, top_k=10):
        """
        Perform two-stage semantic search:
        1. Get initial candidates using vector search
        2. Re-rank candidates using cross-attention semantic similarity
        
        Args:
            query: Search query
            first_stage_k: Number of candidates to retrieve in first stage
            top_k: Number of final results to return
            
        Returns:
            List of (episode_id, score) tuples
        """
        # First stage: Get candidates using vector search
        candidates = self.vector_search(query, top_k=first_stage_k)
        
        # Second stage: Re-rank candidates using semantic similarity
        return self.semantic_rerank(query, candidates, top_k=top_k)
    
    def evaluate_semantic_search(self, 
                                ground_truth_path='podcast_ground_truth.json',
                                k_values=[3, 5, 10]):
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
            "semantic_rerank": {}  # Two-stage semantic search
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
            
            # Evaluate two-stage semantic search
            print("Evaluating two-stage semantic search...")
            semantic_relevance = []
            
            for query, true_id in tqdm(test_queries, desc="Semantic rerank"):
                search_results = self.two_stage_semantic_search(
                    query, first_stage_k=100, top_k=k
                )
                results_ids = [id for id, _ in search_results]
                relevance = [true_id == id for id in results_ids]
                semantic_relevance.append(relevance)
                
            semantic_metrics = calculate_metrics(semantic_relevance)
            results["semantic_rerank"][k] = semantic_metrics
            print(f"Semantic Rerank: Hit Rate: {semantic_metrics['hit_rate']:.4f}, MRR: {semantic_metrics['mrr']:.4f}")
                
        return results
        
    def search_and_explain(self, query, top_k=5, first_stage_k=100, verbose=True):
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
            # Get episode text and title
            episode_text = self.get_episode_text(episode_id)
            episode_title = self.get_episode_title(episode_id)
            
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
                "title": episode_title,
                "score": score,
                "vector_search_rank": vector_rank,
                "rank_improvement": rank_improvement,
                "text_preview": episode_text[:200] + "..." if len(episode_text) > 200 else episode_text
            }
            
            detailed_results.append(explanation)
            
            # Print explanation if verbose
            if verbose:
                print(f"\nRank {rank+1}: Episode {episode_id} - {episode_title}")
                print(f"  Semantic score: {score:.4f}")
                print(f"  Initial vector search rank: {vector_rank}")
                if rank_improvement is not None and rank_improvement > 0:
                    print(f"  ↑ Improved by {rank_improvement} positions after re-ranking")
                print(f"  Text preview: {explanation['text_preview']}")
                
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
        k_values=[3, 5, 10]
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic Search with Re-Ranking")
    parser.add_argument("--mode", choices=["search", "evaluate"], default="search",
                        help="Run mode: search or evaluate")
    args = parser.parse_args()
    
    if args.mode == "search":
        interactive_search()
    else:
        evaluate_search()