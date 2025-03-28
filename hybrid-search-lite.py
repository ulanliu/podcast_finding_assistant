import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os
from huggingface_hub import login
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources if needed
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    print("NLTK resources downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("Will attempt to use tokenization without NLTK resources")

# Function to tokenize text with fallback if NLTK fails
def safe_tokenize(text):
    """Tokenize text with fallback to simple splitting if NLTK fails"""
    try:
        # Try using NLTK tokenizer
        return word_tokenize(text)
    except LookupError:
        # Fallback to simple splitting on whitespace and punctuation
        import re
        return re.findall(r'\w+', text.lower())

# Huggingface login if token available (making this optional)
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
if huggingface_token:
    try:
        login(token=huggingface_token)
        print("Successfully logged in to Huggingface Hub")
    except Exception as e:
        print(f"Warning: Huggingface login failed, but continuing without login: {e}")
        print("Some models may still work with reduced functionality")

class PodcastSearcher:
    def __init__(self, podcast_topic_path='podcast_topic.json', model_name="intfloat/multilingual-e5-base"):
        # Load podcast data
        with open(podcast_topic_path, 'r', encoding='utf-8') as f:
            self.podcast_topic = json.load(f)
        
        # Initialize embedding model with fallback options
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading primary model: {e}")
            # Try fallback models
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
                raise ValueError("Failed to load any embedding model. Please check your internet connection or use a local model.")
        
        # Initialize Qdrant client
        try:
            self.client = QdrantClient(url="http://localhost:6333")
            # Test connection
            self.client.get_collections()
            print("Successfully connected to Qdrant at http://localhost:6333")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            print("Make sure Qdrant is running (check docker-compose.yaml)")
            raise ConnectionError("Cannot connect to Qdrant database")
        
        # Prepare corpus for BM25
        self.corpus = []
        self.episode_ids = []
        self.tokenized_corpus = []
        
        for episode_id, content in self.podcast_topic.items():
            topics = content.get('topic', [])
            song = content.get('song', "")
            
            # Create a text representation
            topics_text = '; '.join(topics) if topics else ""
            full_text = topics_text
            if song and song != "null" and not pd.isna(song):
                full_text += f"; Song: {song}"
            
            if full_text:
                self.corpus.append(full_text)
                self.episode_ids.append(int(episode_id))
                
                # Tokenize for BM25
                tokens = [word.lower() for word in safe_tokenize(full_text) 
                         if word.isalnum() and (not hasattr(stopwords, 'words') or 
                                               word.lower() not in stopwords.words('english'))]
                self.tokenized_corpus.append(tokens)
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Check if collection exists and create if not
        self.collection_name = "podcast_collection"
        self._setup_qdrant()
    
    def _setup_qdrant(self):
        """Set up Qdrant collection if needed"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                print(f"Creating new collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.model.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE
                    )
                )
                
                # Upload vectors
                self._upload_vectors()
            else:
                print(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            print(f"Error setting up Qdrant: {e}")
    
    def _upload_vectors(self):
        """Upload podcast vectors to Qdrant"""
        points = []
        
        for i, (text, episode_id) in enumerate(zip(self.corpus, self.episode_ids)):
            # Embed the text
            embedding = self.model.encode(text)
            
            # Create a point
            points.append(
                models.PointStruct(
                    id=episode_id,
                    vector=embedding.tolist(),
                    payload={
                        'episode': episode_id,
                        'text': text
                    }
                )
            )
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"Uploaded {len(points)} episodes to Qdrant")
    
    def vector_search(self, query, k=5):
        """Perform vector search"""
        query_vector = self.model.encode(query)
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=k
        )
        
        return [(res.id, res.score) for res in search_results]
    
    def keyword_search(self, query, k=5):
        """Perform keyword search using BM25"""
        # Tokenize query
        query_tokens = [w.lower() for w in safe_tokenize(query) 
                       if w.isalnum() and (not hasattr(stopwords, 'words') or 
                                         w.lower() not in stopwords.words('english'))]
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k results
        top_k_indices = np.argsort(scores)[::-1][:k]
        results = [(self.episode_ids[idx], scores[idx]) for idx in top_k_indices]
        
        return results
    
    def hybrid_search(self, query, k=5, alpha=0.7):
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for vector search (1-alpha for keyword search)
        
        Returns:
            List of (episode_id, score) tuples
        """
        # Get vector search results
        vector_results = self.vector_search(query, k=k*2)
        
        # Get keyword search results
        keyword_results = self.keyword_search(query, k=k*2)
        
        # Normalize scores
        max_vector = max([score for _, score in vector_results]) if vector_results else 1.0
        max_keyword = max([score for _, score in keyword_results]) if keyword_results else 1.0
        
        normalized_vector = {id: score/max_vector for id, score in vector_results}
        normalized_keyword = {id: score/max_keyword for id, score in keyword_results}
        
        # Combine scores
        all_ids = set(normalized_vector.keys()) | set(normalized_keyword.keys())
        combined_scores = {}
        
        for id in all_ids:
            vector_score = normalized_vector.get(id, 0)
            keyword_score = normalized_keyword.get(id, 0)
            combined_scores[id] = alpha * vector_score + (1 - alpha) * keyword_score
        
        # Sort and get top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

def evaluate_search(searcher, ground_truth_path='podcast_ground_truth.json', k_values=[3, 5, 10], alpha_values=[0.3, 0.5, 0.7]):
    """
    Evaluate search performance using ground truth data.
    
    Args:
        searcher: PodcastSearcher instance
        ground_truth_path: Path to ground truth data
        k_values: List of k values to evaluate
        alpha_values: List of alpha values for hybrid search
    
    Returns:
        Dictionary with evaluation results
    """
    # Load ground truth data
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    results = {
        "vector": {},
        "keyword": {},
    }
    
    # Add hybrid results with different alpha values
    for alpha in alpha_values:
        results[f"hybrid_alpha_{alpha}"] = {}
    
    # Create test queries from ground truth
    test_queries = []
    for episode_id, episode_data in ground_truth.items():
        words = episode_data.get('word', [])
        for word in words:
            test_queries.append((word, int(episode_id)))
    
    # Evaluate for different k values
    for k in k_values:
        print(f"\nEvaluating for k={k}")
        
        # Evaluate vector search
        print("Evaluating vector search...")
        vector_relevance = []
        for query, true_id in tqdm(test_queries, desc="Vector search"):
            results_ids = [id for id, _ in searcher.vector_search(query, k)]
            relevance = [true_id == id for id in results_ids]
            vector_relevance.append(relevance)
        
        vector_metrics = calculate_metrics(vector_relevance)
        results["vector"][k] = vector_metrics
        print(f"Vector Search: Hit Rate: {vector_metrics['hit_rate']:.4f}, MRR: {vector_metrics['mrr']:.4f}")
        
        # Evaluate keyword search
        print("Evaluating keyword search...")
        keyword_relevance = []
        for query, true_id in tqdm(test_queries, desc="Keyword search"):
            results_ids = [id for id, _ in searcher.keyword_search(query, k)]
            relevance = [true_id == id for id in results_ids]
            keyword_relevance.append(relevance)
        
        keyword_metrics = calculate_metrics(keyword_relevance)
        results["keyword"][k] = keyword_metrics
        print(f"Keyword Search: Hit Rate: {keyword_metrics['hit_rate']:.4f}, MRR: {keyword_metrics['mrr']:.4f}")
        
        # Evaluate hybrid search with different alpha values
        for alpha in alpha_values:
            print(f"Evaluating hybrid search (alpha={alpha})...")
            hybrid_relevance = []
            for query, true_id in tqdm(test_queries, desc=f"Hybrid search (alpha={alpha})"):
                results_ids = [id for id, _ in searcher.hybrid_search(query, k, alpha)]
                relevance = [true_id == id for id in results_ids]
                hybrid_relevance.append(relevance)
            
            hybrid_metrics = calculate_metrics(hybrid_relevance)
            results[f"hybrid_alpha_{alpha}"][k] = hybrid_metrics
            print(f"Hybrid Search (alpha={alpha}): Hit Rate: {hybrid_metrics['hit_rate']:.4f}, MRR: {hybrid_metrics['mrr']:.4f}")
    
    return results

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

# If this script is renamed to something else, this helps with imports
if __name__ == "__main__":
    print("Initializing podcast searcher...")
    searcher = PodcastSearcher()
    
    print("Evaluating search methods...")
    results = evaluate_search(searcher)
    
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
    with open('simple_hybrid_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to simple_hybrid_evaluation_results.json")