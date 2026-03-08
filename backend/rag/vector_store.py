try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.metadata = []
        self.dimension = 384  # MiniLM dimension
        
    def initialize_from_knowledge_base(self, kb_path='rag/knowledge_base.json'):
        """Initialize vector store from knowledge base"""
        logger.info("Initializing vector store from knowledge base")
        
        # Load knowledge base documents
        with open(kb_path, 'r') as f:
            kb_data = json.load(f)
        
        documents = []
        metadatas = []
        
        for category, items in kb_data.items():
            for item in items:
                documents.append(item['content'])
                metadatas.append({
                    'category': category,
                    'source': item.get('source', 'knowledge_base'),
                    'timestamp': datetime.now().isoformat()
                })
        
        self.add_documents(documents, metadatas)
        logger.info(f"Added {len(documents)} documents to vector store")
        
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents to vector store"""
        if not documents:
            return
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)
        
        # Initialize FAISS index if needed
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in documents])
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), min(k, self.index.ntotal))
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity_score': float(1 / (1 + distances[0][i]))  # Convert distance to similarity
                })
        
        return results
    
    def save(self, path: str = 'rag/vector_store'):
        """Save vector store to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, f'{path}/index.faiss')
        
        # Save documents and metadata
        with open(f'{path}/documents.pkl', 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str = 'rag/vector_store'):
        """Load vector store from disk"""
        # Load FAISS index
        index_path = f'{path}/index.faiss'
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # Load documents and metadata
        docs_path = f'{path}/documents.pkl'
        if os.path.exists(docs_path):
            with open(docs_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
        
        logger.info(f"Vector store loaded from {path}")