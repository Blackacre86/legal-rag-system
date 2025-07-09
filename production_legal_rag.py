#!/usr/bin/env python3
"""
Production-Ready Legal RAG System
Optimized for Massachusetts Criminal Law Documents
"""

import os
import json
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime

# Core RAG components
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
from transformers import AutoTokenizer

# Document processing
import fitz  # PyMuPDF
import tiktoken

# Configuration and utilities
from dotenv import load_dotenv
import hashlib
import pickle

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LegalChunk:
    """Enhanced chunk with legal metadata"""
    id: str
    content: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    citations: List[str] = None
    legal_concepts: List[str] = None
    document_type: str = "unknown"
    source_file: str = ""
    page_number: int = 0
    confidence_score: float = 0.0
    token_count: int = 0
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.legal_concepts is None:
            self.legal_concepts = []
        if not self.token_count:
            self.token_count = self._count_tokens()
    
    def _count_tokens(self) -> int:
        """Count tokens using tiktoken"""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(self.content))
        except:
            # Fallback estimation
            return len(self.content.split()) * 1.3

@dataclass 
class RetrievalResult:
    """Result from RAG retrieval"""
    chunk: LegalChunk
    similarity_score: float
    rank: int
    retrieval_method: str

class ProductionLegalRAG:
    """Production-ready RAG system optimized for legal documents"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 openai_api_key: str = None,
                 cache_dir: str = "./cache"):
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize OpenAI
        openai.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize vector index
        self.index = None
        self.chunks: List[LegalChunk] = []
        
        # Legal processing patterns
        self.legal_patterns = self._initialize_legal_patterns()
        self.legal_concepts = self._initialize_legal_concepts()
        
        logger.info(f"Initialized Legal RAG with {model_name}")

    def _initialize_legal_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize patterns for legal document processing"""
        return {
            'chapter': re.compile(r'^CHAPTER\s+(\d+)[:\-]?\s*(.+)$', re.IGNORECASE | re.MULTILINE),
            'section': re.compile(r'^(?:Section|§)\s*([\d.]+)[:\-]?\s*(.+)$', re.IGNORECASE | re.MULTILINE),
            'subsection': re.compile(r'^\s*\(([a-z]|\d+)\)\s*(.+)$', re.MULTILINE),
            'mgl_citation': re.compile(r'(?:M\.?G\.?L\.?|General Laws?)\s*c(?:hapter|\.)?\s*(\d+)[,\s]*(?:§|s(?:ection|\.))\s*([\d\.]+)', re.IGNORECASE),
            'case_citation': re.compile(r'(\w+(?:\s+\w+)*)\s+v\.\s+(\w+(?:\s+\w+)*),\s*(\d+)\s+Mass\.\s*(\d+)(?:\s*\((\d{4})\))?', re.IGNORECASE),
            'federal_case': re.compile(r'(\w+(?:\s+\w+)*)\s+v\.\s+(\w+(?:\s+\w+)*),\s*(\d+)\s+U\.S\.\s*(\d+)', re.IGNORECASE),
            'definition': re.compile(r'"([^"]+)"\s+(?:means?|is defined as)', re.IGNORECASE),
            'elements_start': re.compile(r'(?:elements?|requirements?|conditions?).*?(?:are|include)?:?\s*$', re.IGNORECASE),
            'numbered_list': re.compile(r'^\s*(\d+)\.?\s+(.+)$', re.MULTILINE),
            'offense_class': re.compile(r'(?:Class [A-E]|felony|misdemeanor)', re.IGNORECASE),
            'penalty': re.compile(r'(?:fine|imprisonment|jail|prison|penalty).*?(?:\$[\d,]+|\d+\s+(?:days?|months?|years?))', re.IGNORECASE)
        }

    def _initialize_legal_concepts(self) -> Dict[str, List[str]]:
        """Initialize legal concept keywords"""
        return {
            'miranda_rights': ['Miranda', 'right to remain silent', 'right to counsel', 'waiver', 'custody', 'interrogation'],
            'probable_cause': ['probable cause', 'PC', 'reasonable grounds', 'reasonable belief', 'articulable facts'],
            'search_warrant': ['warrant', 'search authorization', 'judicial approval', 'magistrate', 'search warrant'],
            'automobile_exception': ['automobile exception', 'motor vehicle exception', 'Carroll doctrine', 'mobility'],
            'terry_stop': ['Terry stop', 'investigatory stop', 'reasonable suspicion', 'frisk', 'stop and frisk'],
            'use_of_force': ['use of force', 'deadly force', 'reasonable force', 'excessive force', 'force continuum'],
            'arrest': ['arrest', 'custodial arrest', 'arrest warrant', 'warrantless arrest', 'custody'],
            'oui_dui': ['OUI', 'DUI', 'operating under influence', 'drunk driving', 'breathalyzer', 'field sobriety'],
            'consent_search': ['consent', 'voluntary consent', 'consent to search', 'third party consent'],
            'exigent_circumstances': ['exigent circumstances', 'emergency', 'hot pursuit', 'immediate danger', 'destruction of evidence'],
            'fourth_amendment': ['Fourth Amendment', 'unreasonable search', 'seizure', 'expectation of privacy'],
            'fifth_amendment': ['Fifth Amendment', 'self-incrimination', 'double jeopardy', 'due process'],
            'sixth_amendment': ['Sixth Amendment', 'right to counsel', 'confrontation', 'speedy trial'],
            'exclusionary_rule': ['exclusionary rule', 'fruit of poisonous tree', 'suppression', 'evidence exclusion'],
            'plain_view': ['plain view', 'plain sight', 'immediately apparent', 'inadvertent discovery'],
            'inventory_search': ['inventory search', 'impoundment', 'vehicle inventory', 'standardized procedure']
        }

    def process_document(self, pdf_path: str, document_type: str = "scheft_manual") -> List[LegalChunk]:
        """Process legal document with enhanced chunking"""
        
        logger.info(f"Processing document: {pdf_path}")
        
        # Check cache
        cache_key = self._get_cache_key(pdf_path)
        cached_chunks = self._load_from_cache(cache_key)
        if cached_chunks:
            logger.info("Loaded chunks from cache")
            return cached_chunks
        
        # Extract text with structure
        doc = fitz.open(pdf_path)
        chunks = []
        
        current_chapter = ""
        current_section = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            if not text.strip():
                continue
            
            # Process page in sections
            page_chunks = self._process_page_text(
                text, 
                page_num + 1, 
                current_chapter, 
                current_section,
                document_type,
                os.path.basename(pdf_path)
            )
            
            # Update context for next page
            for chunk in page_chunks:
                if chunk.chapter:
                    current_chapter = chunk.chapter
                if chunk.section:
                    current_section = chunk.section
            
            chunks.extend(page_chunks)
        
        doc.close()
        
        # Post-process chunks
        chunks = self._post_process_chunks(chunks)
        
        # Cache results
        self._save_to_cache(cache_key, chunks)
        
        logger.info(f"Created {len(chunks)} legal chunks")
        return chunks

    def _process_page_text(self, text: str, page_num: int, current_chapter: str, 
                          current_section: str, document_type: str, source_file: str) -> List[LegalChunk]:
        """Process text from a single page"""
        
        chunks = []
        
        # Split by strong structural breaks
        sections = self._split_by_legal_structure(text)
        
        for section_text in sections:
            section_text = section_text.strip()
            if len(section_text) < 50:  # Skip very short sections
                continue
            
            # Extract metadata
            chapter = self._extract_chapter(section_text) or current_chapter
            section = self._extract_section(section_text) or current_section
            
            # Create semantic chunks
            section_chunks = self._create_legal_chunks(
                section_text,
                chapter,
                section,
                page_num,
                document_type,
                source_file
            )
            
            chunks.extend(section_chunks)
        
        return chunks

    def _split_by_legal_structure(self, text: str) -> List[str]:
        """Split text at natural legal document boundaries"""
        
        # First split on chapters
        chapter_pattern = r'\n(?=CHAPTER\s+\d+)'
        sections = re.split(chapter_pattern, text, flags=re.IGNORECASE)
        
        # Further split on major sections
        result = []
        for section in sections:
            # Split on section headers
            section_pattern = r'\n(?=Section\s+[\d.]+)'
            subsections = re.split(section_pattern, section, flags=re.IGNORECASE)
            result.extend(subsections)
        
        return [s.strip() for s in result if s.strip()]

    def _create_legal_chunks(self, text: str, chapter: str, section: str, 
                           page_num: int, document_type: str, source_file: str) -> List[LegalChunk]:
        """Create chunks that preserve legal semantics"""
        
        chunks = []
        
        # If text is reasonably sized and semantically coherent, keep as one chunk
        if len(text) <= 1200 and self._is_coherent_legal_unit(text):
            chunk = self._create_chunk(
                text, chapter, section, page_num, document_type, source_file
            )
            chunks.append(chunk)
            return chunks
        
        # For longer text, split more carefully
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk_text = ""
        
        for para in paragraphs:
            # Test adding this paragraph
            test_text = current_chunk_text + "\n\n" + para if current_chunk_text else para
            
            # Check if we should break here
            should_break = (
                len(test_text) > 1000 and 
                self._should_break_at_paragraph(current_chunk_text, para)
            )
            
            if should_break and current_chunk_text:
                # Create chunk from current text
                chunk = self._create_chunk(
                    current_chunk_text, chapter, section, page_num, document_type, source_file
                )
                chunks.append(chunk)
                current_chunk_text = para
            else:
                current_chunk_text = test_text
        
        # Create final chunk
        if current_chunk_text:
            chunk = self._create_chunk(
                current_chunk_text, chapter, section, page_num, document_type, source_file
            )
            chunks.append(chunk)
        
        return chunks

    def _create_chunk(self, text: str, chapter: str, section: str, page_num: int,
                     document_type: str, source_file: str) -> LegalChunk:
        """Create a single legal chunk with metadata"""
        
        chunk_id = hashlib.md5(f"{source_file}_{page_num}_{text[:100]}".encode()).hexdigest()[:12]
        
        return LegalChunk(
            id=chunk_id,
            content=text,
            chapter=chapter,
            section=section,
            citations=self._extract_citations(text),
            legal_concepts=self._extract_legal_concepts(text),
            document_type=document_type,
            source_file=source_file,
            page_number=page_num,
            confidence_score=self._calculate_chunk_quality(text)
        )

    def _is_coherent_legal_unit(self, text: str) -> bool:
        """Check if text represents a coherent legal concept"""
        
        # Contains complete definition
        if self.legal_patterns['definition'].search(text):
            return True
        
        # Contains complete elements list
        elements_match = self.legal_patterns['elements_start'].search(text)
        numbered_items = self.legal_patterns['numbered_list'].findall(text)
        if elements_match and len(numbered_items) >= 2:
            return True
        
        # Single focused legal concept
        concepts = self._extract_legal_concepts(text)
        citations = self._extract_citations(text)
        
        return len(concepts) <= 2 and len(citations) > 0

    def _should_break_at_paragraph(self, current: str, new_para: str) -> bool:
        """Determine if we should break chunk at this paragraph"""
        
        # Don't break in middle of numbered list
        if (self.legal_patterns['elements_start'].search(current) and 
            self.legal_patterns['numbered_list'].search(new_para)):
            return False
        
        # Don't break definition from examples
        if ('means' in current.lower() or 'defined as' in current.lower()):
            if 'example' in new_para.lower() or 'for instance' in new_para.lower():
                return False
        
        # Break on major topic shift
        current_concepts = set(self._extract_legal_concepts(current))
        new_concepts = set(self._extract_legal_concepts(new_para))
        
        # If no overlap in legal concepts, consider breaking
        if current_concepts and new_concepts and not (current_concepts & new_concepts):
            return True
        
        return False

    def _extract_chapter(self, text: str) -> Optional[str]:
        """Extract chapter number from text"""
        match = self.legal_patterns['chapter'].search(text)
        return match.group(1) if match else None

    def _extract_section(self, text: str) -> Optional[str]:
        """Extract section number from text"""
        match = self.legal_patterns['section'].search(text)
        return match.group(1) if match else None

    def _extract_citations(self, text: str) -> List[str]:
        """Extract legal citations"""
        citations = []
        
        # MGL citations
        for match in self.legal_patterns['mgl_citation'].finditer(text):
            citations.append(f"M.G.L. c. {match.group(1)}, § {match.group(2)}")
        
        # Massachusetts case citations
        for match in self.legal_patterns['case_citation'].finditer(text):
            plaintiff, defendant, volume, page, year = match.groups()
            citation = f"{plaintiff} v. {defendant}, {volume} Mass. {page}"
            if year:
                citation += f" ({year})"
            citations.append(citation)
        
        # Federal case citations
        for match in self.legal_patterns['federal_case'].finditer(text):
            plaintiff, defendant, volume, page = match.groups()
            citations.append(f"{plaintiff} v. {defendant}, {volume} U.S. {page}")
        
        return list(set(citations))  # Remove duplicates

    def _extract_legal_concepts(self, text: str) -> List[str]:
        """Extract legal concepts from text"""
        concepts = []
        text_lower = text.lower()
        
        for concept, keywords in self.legal_concepts.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts

    def _calculate_chunk_quality(self, text: str) -> float:
        """Calculate quality score for legal chunk"""
        score = 0.0
        
        # Has legal citations
        citations = self._extract_citations(text)
        if citations:
            score += 0.25 + min(len(citations) * 0.05, 0.15)  # Bonus for multiple citations
        
        # Has legal concepts
        concepts = self._extract_legal_concepts(text)
        if concepts:
            score += 0.2 + min(len(concepts) * 0.03, 0.1)
        
        # Appropriate length (300-1000 tokens ideal)
        token_count = len(text.split()) * 1.3  # Rough estimate
        if 300 <= token_count <= 1000:
            score += 0.2
        elif 200 <= token_count <= 1200:
            score += 0.1
        
        # Has legal structure
        if (self.legal_patterns['numbered_list'].search(text) or 
            self.legal_patterns['definition'].search(text) or
            self.legal_patterns['elements_start'].search(text)):
            score += 0.15
        
        # Has penalties/classifications
        if (self.legal_patterns['offense_class'].search(text) or
            self.legal_patterns['penalty'].search(text)):
            score += 0.1
        
        # Completeness (not fragment)
        sentences = text.count('.') + text.count('!') + text.count('?')
        if sentences >= 2:
            score += 0.1
        
        return min(score, 1.0)

    def _post_process_chunks(self, chunks: List[LegalChunk]) -> List[LegalChunk]:
        """Post-process chunks for quality and consistency"""
        
        processed_chunks = []
        
        for chunk in chunks:
            # Skip very low quality chunks
            if chunk.confidence_score < 0.3:
                continue
            
            # Skip very short chunks without legal content
            if len(chunk.content) < 100 and not (chunk.citations or chunk.legal_concepts):
                continue
            
            # Merge very short high-quality chunks with next chunk if appropriate
            if (len(chunk.content) < 200 and 
                chunk.confidence_score > 0.7 and 
                processed_chunks and
                processed_chunks[-1].chapter == chunk.chapter and
                processed_chunks[-1].section == chunk.section):
                
                # Merge with previous chunk
                prev_chunk = processed_chunks[-1]
                merged_content = prev_chunk.content + "\n\n" + chunk.content
                merged_citations = list(set(prev_chunk.citations + chunk.citations))
                merged_concepts = list(set(prev_chunk.legal_concepts + chunk.legal_concepts))
                
                # Update previous chunk
                prev_chunk.content = merged_content
                prev_chunk.citations = merged_citations
                prev_chunk.legal_concepts = merged_concepts
                prev_chunk.confidence_score = self._calculate_chunk_quality(merged_content)
                prev_chunk.token_count = 0  # Will be recalculated
                
                continue
            
            processed_chunks.append(chunk)
        
        logger.info(f"Post-processing: {len(chunks)} -> {len(processed_chunks)} chunks")
        return processed_chunks

    def build_index(self, chunks: List[LegalChunk]):
        """Build FAISS index from legal chunks"""
        
        logger.info("Building vector index...")
        self.chunks = chunks
        
        # Generate embeddings
        contents = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
        
        # Store embeddings in chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))
        
        logger.info(f"Index built with {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 10, filters: Dict = None) -> List[RetrievalResult]:
        """Retrieve relevant chunks for query"""
        
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search vector index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)
        
        # Create results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
                
            chunk = self.chunks[idx]
            
            # Apply filters
            if filters and not self._passes_filters(chunk, filters):
                continue
            
            results.append(RetrievalResult(
                chunk=chunk,
                similarity_score=float(score),
                rank=len(results),
                retrieval_method="semantic"
            ))
            
            if len(results) >= top_k:
                break
        
        return results

    def generate_response(self, query: str, retrieved_chunks: List[RetrievalResult], 
                         max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response using OpenAI with retrieved context"""
        
        # Build context from retrieved chunks
        context_parts = []
        citations = set()
        
        for result in retrieved_chunks[:5]:  # Use top 5 results
            chunk = result.chunk
            context_parts.append(f"[Source: {chunk.source_file}, Chapter {chunk.chapter}, Section {chunk.section}]\n{chunk.content}")
            citations.update(chunk.citations)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Construct prompt
        system_prompt = """You are an expert Massachusetts criminal law assistant. Answer questions based ONLY on the provided legal sources. Always cite specific sources using the format provided in the context. Be precise, accurate, and practical for law enforcement officers.

CRITICAL RULES:
1. Only use information from the provided context
2. Always cite sources using the exact format: [Source: filename, Chapter X, Section Y]
3. Include relevant case citations and statutes
4. If the context doesn't contain enough information, say so explicitly
5. Use clear, practical language for police officers"""

        user_prompt = f"""Context from Massachusetts Criminal Law sources:

{context}

Question: {query}

Provide a comprehensive answer based on the provided sources. Include specific citations for every legal claim."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                presence_penalty=0.1
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "source_file": result.chunk.source_file,
                        "chapter": result.chunk.chapter,
                        "section": result.chunk.section,
                        "content_preview": result.chunk.content[:200] + "...",
                        "similarity_score": result.similarity_score,
                        "citations": result.chunk.citations
                    }
                    for result in retrieved_chunks[:5]
                ],
                "legal_citations": list(citations),
                "query_id": hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()[:12],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "error": str(e),
                "answer": "I apologize, but I encountered an error generating a response. Please try again.",
                "sources": [],
                "legal_citations": [],
                "query_id": None,
                "timestamp": datetime.now().isoformat()
            }

    def query(self, query: str, top_k: int = 10, filters: Dict = None) -> Dict[str, Any]:
        """Complete RAG query pipeline"""
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Retrieve relevant chunks
        retrieved = self.retrieve(query, top_k, filters)
        
        if not retrieved:
            return {
                "answer": "I couldn't find relevant information in the legal sources for your question.",
                "sources": [],
                "legal_citations": [],
                "query_id": None,
                "timestamp": datetime.now().isoformat()
            }
        
        # Generate response
        response = self.generate_response(query, retrieved)
        
        return response

    def _passes_filters(self, chunk: LegalChunk, filters: Dict) -> bool:
        """Check if chunk passes filters"""
        
        if 'chapter' in filters and chunk.chapter != filters['chapter']:
            return False
        
        if 'section' in filters and chunk.section != filters['section']:
            return False
        
        if 'legal_concepts' in filters:
            required_concepts = set(filters['legal_concepts'])
            chunk_concepts = set(chunk.legal_concepts)
            if not (required_concepts & chunk_concepts):
                return False
        
        if 'min_quality' in filters and chunk.confidence_score < filters['min_quality']:
            return False
        
        return True

    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key for file"""
        stat = os.stat(file_path)
        return hashlib.md5(f"{file_path}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()

    def _save_to_cache(self, cache_key: str, chunks: List[LegalChunk]):
        """Save chunks to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(chunks, f)

    def _load_from_cache(self, cache_key: str) -> Optional[List[LegalChunk]]:
        """Load chunks from cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def save_index(self, path: str):
        """Save index and chunks to disk"""
        save_path = Path(path)
        save_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save chunks
        chunks_data = [asdict(chunk) for chunk in self.chunks]
        with open(save_path / "chunks.json", 'w') as f:
            json.dump(chunks_data, f, indent=2, default=str)
        
        logger.info(f"Index saved to {path}")

    def load_index(self, path: str):
        """Load index and chunks from disk"""
        load_path = Path(path)
        
        # Load FAISS index
        index_file = load_path / "index.faiss"
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))
        
        # Load chunks
        chunks_file = load_path / "chunks.json"
        if chunks_file.exists():
            with open(chunks_file, 'r') as f:
                chunks_data = json.load(f)
            
            self.chunks = []
            for chunk_dict in chunks_data:
                # Convert back to LegalChunk object
                chunk = LegalChunk(**chunk_dict)
                self.chunks.append(chunk)
        
        logger.info(f"Index loaded from {path}")

def main():
    """Example usage of the production legal RAG system"""
    
    # Initialize system
    rag = ProductionLegalRAG(
        model_name="all-MiniLM-L6-v2",  # or "sentence-transformers/all-mpnet-base-v2" for better quality
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Process Scheft manuals
    documents = [
        ("scheft_criminal_law_2025.pdf", "criminal_law"),
        ("scheft_criminal_procedure_2025.pdf", "criminal_procedure"),
        ("scheft_motor_vehicle_law_2025.pdf", "motor_vehicle"),
        ("scheft_juvenile_law_2025.pdf", "juvenile_law")
    ]
    
    all_chunks = []
    for pdf_path, doc_type in documents:
        if os.path.exists(pdf_path):
            chunks = rag.process_document(pdf_path, doc_type)
            all_chunks.extend(chunks)
            print(f"Processed {pdf_path}: {len(chunks)} chunks")
    
    # Build index
    rag.build_index(all_chunks)
    
    # Save for future use
    rag.save_index("./legal_rag_index")
    
    # Test queries
    test_queries = [
        "What are the elements of OUI in Massachusetts?",
        "When can police conduct a protective sweep?",
        "What constitutes a valid Miranda waiver?",
        "Explain the automobile exception to warrant requirement",
        "What are the requirements for a Terry stop?"
    ]
    
    print("\n" + "="*60)
    print("TESTING LEGAL RAG SYSTEM")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        result = rag.query(query, top_k=5)
        
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Sources: {len(result['sources'])}")
        print(f"Citations: {result['legal_citations'][:3]}")
        print()

if __name__ == "__main__":
    main()