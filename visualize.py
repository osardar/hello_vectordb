import argparse
import matplotlib.pyplot as plt
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import seaborn as sns

def get_all_vectors(client: QdrantClient, collection_name: str, limit: int = 1000):
    """Retrieve all vectors and their payloads from the collection"""
    results = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=True
    )
    
    vectors = []
    sentences = []
    topics = []
    
    for point in results[0]:
        vectors.append(point.vector)
        sentences.append(point.payload['sentence'])
        topics.append(point.payload['topic'])
    
    return np.array(vectors), sentences, topics

def visualize_clusters(vectors, sentences, topics):
    """Visualize sentence clusters using t-SNE dimensionality reduction"""
    
    # t-SNE for dimensionality reduction
    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1))
    coords = reducer.fit_transform(vectors)
    title = "t-SNE Clustering of Sentence Embeddings"
    
    # Create unique colors for each topic
    unique_topics = list(set(topics))
    colors = sns.color_palette("husl", len(unique_topics))
    topic_to_color = {topic: colors[i] for i, topic in enumerate(unique_topics)}
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each point with its topic color
    for i, (x, y) in enumerate(coords):
        topic = topics[i]
        color = topic_to_color[topic]
        plt.scatter(x, y, c=[color], alpha=0.7, s=50)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=topic_to_color[topic], 
                                 markersize=10, label=topic) 
                      for topic in unique_topics]
    plt.legend(handles=legend_elements, title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    
    return coords

def interactive_plot(vectors, sentences, topics, coords):
    """Create an interactive plot with hover information"""
    try:
        import plotly.express as px
        import pandas as pd
        
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'sentence': sentences,
            'topic': topics
        })
        
        fig = px.scatter(df, x='x', y='y', color='topic', 
                        hover_data=['sentence'], 
                        title="Interactive Sentence Clusters")
        fig.show()
        
    except ImportError:
        print("Plotly not installed. Install with: uv add plotly")
        print("Showing static plot instead...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--collection_name", type=str, default="collection_0")

    parser.add_argument("--interactive", action="store_true", help="Use interactive plotly visualization")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of vectors to retrieve")
    
    args = parser.parse_args()
    
    # Connect to Qdrant
    client = QdrantClient(host=args.host, port=args.port)
    
    print(f"Retrieving vectors from collection '{args.collection_name}'...")
    vectors, sentences, topics = get_all_vectors(client, args.collection_name, args.limit)
    
    print(f"Retrieved {len(vectors)} vectors")
    print(f"Unique topics: {set(topics)}")
    
    # Visualize clusters
    print("Creating t-SNE visualization...")
    coords = visualize_clusters(vectors, sentences, topics)
    
    if args.interactive:
        interactive_plot(vectors, sentences, topics, coords)
    
    plt.savefig("sentence_clusters_tsne.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 