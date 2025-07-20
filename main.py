import argparse
import csv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer


def init_qdrant_client(host: str, port: int):
    return QdrantClient(host=host, port=port)


def init_qdrant_collection(client: QdrantClient, collection_name: str):
    # Check if collection already exists
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]
    
    if collection_name not in collection_names:
        return client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    else:
        # Check if existing collection has correct dimensions
        collection_info = client.get_collection(collection_name)
        current_size = collection_info.config.params.vectors.size
        
        if current_size != 384:
            print(f"Collection '{collection_name}' exists but has wrong dimensions ({current_size}). Deleting and recreating...")
            client.delete_collection(collection_name)
            return client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        else:
            print(f"Collection '{collection_name}' already exists with correct dimensions")
            return None


def transform_to_vector(model: SentenceTransformer, text: str):
    return model.encode(text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--collection_name", type=str, default="collection_0")
    parser.add_argument(
        "--csv", type=str, help="Path to input CSV file"
    )
    parser.add_argument(
        "--query", type=str, help="Search query sentence"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = init_qdrant_client(args.host, args.port)
    collection = init_qdrant_collection(client, args.collection_name)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if collection:
        print(f"Collection '{args.collection_name}' created successfully")
    else:
        print(f"Using existing collection '{args.collection_name}'")

    if args.csv:
        print(f"CSV file specified: {args.csv}")
        with open(args.csv, "r") as file:
            csv_reader = csv.reader(file)
            # Skip header if it exists
            next(csv_reader, None)
            
            # Read all rows and create vectors
            rows = list(csv_reader)
            vectors = map(lambda row: transform_to_vector(model, row[0]), rows)

            client.upsert(
                collection_name=args.collection_name,
                points=[
                    PointStruct(
                        id=idx,
                        vector=vector.tolist(),
                        payload={"sentence": row[0], "topic": row[1]},
                    )
                    for idx, (row, vector) in enumerate(zip(rows, vectors))
                ],
            )
            print(f"Inserted {len(rows)} vectors into collection")

    # Handle search query
    if args.query:
        print(f"Searching for: {args.query}")
        query_vector = transform_to_vector(model, args.query)
        
        search_results = client.search(
            collection_name=args.collection_name,
            query_vector=query_vector.tolist(),
            limit=5
        )
        
        print("\nSearch results:")
        for i, result in enumerate(search_results, 1):
            print(f"{i}. Score: {result.score:.4f}")
            print(f"   Sentence: {result.payload['sentence']}")
            print(f"   Topic: {result.payload['topic']}")
            print()

if __name__ == "__main__":
    main()
