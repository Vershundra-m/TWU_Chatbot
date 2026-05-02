import pinecone
from pinecone import ServerlessSpec
import config
import time

class VectorStore:
    def __init__(self):
        self.pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.INDEX_NAME
        self.index = None
    
    def create_index(self):
        """Create a new index (deletes existing)"""
        
        # Delete if exists 
        if self.index_name in self.pc.list_indexes().names():
            self.pc.delete_index(self.index_name)
            time.sleep(5)
        
        # Create new index
        self.pc.create_index(
            name=self.index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
        print(f" Index '{self.index_name}' ready")
    
    def connect_index(self):
        """Connect to existing index """
        
        if self.index_name not in self.pc.list_indexes().names():
            print(f" Index '{self.index_name}' does not exist. Call create_index() first.")
            return False
        
        self.index = self.pc.Index(self.index_name)
        print(f" Connected to existing index '{self.index_name}'")
        return True
    
    def upload_chunks(self, chunks):
        """Upload chunks to Pinecone"""
        
        if not self.index:
            print(" No index connection. Call connect_index() or create_index() first.")
            return
        
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self.pc.inference.embed(
                model=config.EMBEDDING_MODEL,
                inputs=[chunk],
                parameters={"input_type": "passage"}
            )
            
            vectors.append({
                'id': f'chunk_{i:03d}',
                'values': embedding[0].values,
                'metadata': {'text': chunk}
            })
        
        # Upload in batches
        batch_size = 10
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        print(f" Chunks {len(chunks)} Uploaded")
    
    def query(self, query_text, top_k=config.TOP_K_RESULTS):
        """Search for relevant chunks"""
        
        if not self.index:
            print(" No index connection. Call connect_index() first.")
            return []
        
        # Generate query embedding
        embedding = self.pc.inference.embed(
            model=config.EMBEDDING_MODEL,
            inputs=[query_text],
            parameters={"input_type": "query"}
        )
        
        # Search
        results = self.index.query(
            vector=embedding[0].values,
            top_k=top_k,
            include_metadata=True
        )
        
        chunks = []
        for match in results['matches']:
            chunks.append({
                'text': match['metadata']['text'],
                'score': match['score']
            })
        
        return chunks
