"""
Vector Store Module.
Indexes real-world context notes into ChromaDB for semantic retrieval (RAG).
Includes metadata filtering (date, corridor) and semantic search.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.config import settings


class ContextNotesVectorStore:
    """Manages ingestion, embedding, and semantic retrieval of context notes."""

    def __init__(
        self,
        notes_path: Union[str, Path] = settings.CONTEXT_NOTES_PATH,
        persist_dir: Optional[Union[str, Path]] = settings.CHROMA_PERSIST_DIR,
        collection_name: str = settings.CHROMA_COLLECTION_NAME,
    ):
        self.notes_path = Path(notes_path)
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.collection_name = collection_name
        
        # Initialize ChromaDB client (persistent if dir specified, else ephemeral)
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        else:
            self.client = chromadb.Client()
            
        self.collection = self._init_collection()

    def _init_collection(self):
        """Initializes or resets the collection and indexes all notes."""
        # Get or create collection
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # If collection is empty, index notes from CSV
        if collection.count() == 0 and self.notes_path.exists():
            self._index_notes(collection)
            
        return collection

    def _index_notes(self, collection):
        """Reads context_notes.csv and adds documents to ChromaDB."""
        df_notes = pd.read_csv(self.notes_path)
        
        documents = []
        metadatas = []
        ids = []
        
        for _, row in df_notes.iterrows():
            note_id = str(row["note_id"]).strip()
            date = str(row["date"]).strip()
            applies_to = str(row["applies_to"]).strip()
            note_text = str(row["note"]).strip()
            
            # Combine corridor and note text for rich semantic embedding
            doc_content = f"Date: {date} | Route: {applies_to} | {note_text}"
            
            documents.append(doc_content)
            metadatas.append({
                "note_id": note_id,
                "date": date,
                "applies_to": applies_to,
                "raw_note": note_text
            })
            ids.append(note_id)
            
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for semantically matching context notes.
        Returns a list of dicts with note_id, date, applies_to, note, distance.
        """
        query_args = {
            "query_texts": [query_text],
            "n_results": n_results
        }
        if where_filter:
            query_args["where"] = where_filter
            
        results = self.collection.query(**query_args)
        
        formatted = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            for i, note_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                doc = results["documents"][0][i] if results.get("documents") else ""
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                
                formatted.append({
                    "note_id": note_id,
                    "date": metadata.get("date", ""),
                    "applies_to": metadata.get("applies_to", ""),
                    "raw_note": metadata.get("raw_note", doc),
                    "document": doc,
                    "distance": distance
                })
        return formatted

    def get_all_notes(self) -> List[Dict[str, Any]]:
        """Returns all indexed context notes."""
        df_notes = pd.read_csv(self.notes_path)
        return df_notes.to_dict(orient="records")


def get_vector_store() -> ContextNotesVectorStore:
    """Singleton getter for vector store."""
    return ContextNotesVectorStore()
