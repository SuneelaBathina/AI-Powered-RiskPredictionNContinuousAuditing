try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

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
        self.embeddings = None
        self.documents = []
        self.metadata = []
        self.dimension = 384  # MiniLM dimension
        self.use_faiss = FAISS_AVAILABLE
        if not self.use_faiss:
            logger.warning("FAISS not available; using NumPy fallback for similarity search")
        
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
        
        embeddings = self.embedding_model.encode(documents).astype('float32')
        
        if self.use_faiss:
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
        else:
            if self.embeddings is None:
                self.embeddings = np.empty((0, self.dimension), dtype='float32')
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.documents.extend(documents)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in documents])
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.use_faiss:
            if self.index is None or self.index.ntotal == 0:
                return []
            query_embedding = self.embedding_model.encode([query]).astype('float32')
            distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'content': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'similarity_score': float(1 / (1 + distances[0][i]))
                    })
            return results
        
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return []
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
        best_indices = np.argsort(distances)[:min(k, len(distances))]
        results = []
        for idx in best_indices:
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity_score': float(1 / (1 + distances[idx]))
            })
        return results
    
    def save(self, path: str = 'rag/vector_store'):
        """Save vector store to disk"""
        os.makedirs(path, exist_ok=True)
        
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, f'{path}/index.faiss')

        if self.embeddings is not None:
            np.save(f'{path}/embeddings.npy', self.embeddings)
        
        with open(f'{path}/documents.pkl', 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str = 'rag/vector_store'):
        """Load vector store from disk"""
        index_path = f'{path}/index.faiss'
        if os.path.exists(index_path) and self.use_faiss:
            self.index = faiss.read_index(index_path)
        
        embeddings_path = f'{path}/embeddings.npy'
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path).astype('float32')
        
        docs_path = f'{path}/documents.pkl'
        if os.path.exists(docs_path):
            with open(docs_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
        
        if not self.use_faiss and self.embeddings is None and self.documents:
            self.embeddings = self.embedding_model.encode(self.documents).astype('float32')

        logger.info(f"Vector store loaded from {path}")
