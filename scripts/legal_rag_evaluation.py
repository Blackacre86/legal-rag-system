#!/usr/bin/env python3
"""
Legal RAG Evaluation & Optimization System
Designed specifically for Massachusetts Criminal Law documents
"""

import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from collections import defaultdict

# Core imports for RAG pipeline
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModel
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalChunk:
    """Enhanced chunk with legal metadata"""
    content: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    citations: List[str] = None
    legal_concepts: List[str] = None
    document_type: str = "unknown"
    page_number: int = 0
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.legal_concepts is None:
            self.legal_concepts = []

class LegalDocumentProcessor:
    """Advanced processor for Massachusetts legal documents"""
    
    def __init__(self):
        # Legal patterns for Massachusetts documents
        self.patterns = {
            'chapter': re.compile(r'^CHAPTER\s+(\d+)[:\-]?\s*(.+)$', re.IGNORECASE | re.MULTILINE),
            'section': re.compile(r'^(?:Section|§)\s*([\d.]+)[:\-]?\s*(.+)$', re.IGNORECASE | re.MULTILINE),
            'subsection': re.compile(r'^\s*\(([a-z]|\d+)\)\s*(.+)$', re.MULTILINE),
            'mgl_citation': re.compile(r'(?:M\.?G\.?L\.?|General Laws?)\s*c(?:hapter|\.)?\s*(\d+)[,\s]*(?:§|s(?:ection|\.))\s*([\d\.]+)', re.IGNORECASE),
            'case_citation': re.compile(r'(\w+(?:\s+\w+)*)\s+v\.\s+(\w+(?:\s+\w+)*),\s*(\d+)\s+Mass\.\s*(\d+)(?:\s*\((\d{4})\))?', re.IGNORECASE),
            'definition': re.compile(r'"([^"]+)"\s+(?:means?|is defined as)', re.IGNORECASE),
            'elements_start': re.compile(r'(?:elements?|requirements?|conditions?).*?(?:are|include)?:?\s*$', re.IGNORECASE),
            'numbered_list': re.compile(r'^\s*(\d+)\.?\s+(.+)$', re.MULTILINE)
        }
        
        # Legal concepts to track
        self.legal_concepts = {
            'miranda_rights': ['Miranda', 'right to remain silent', 'right to counsel', 'waiver'],
            'probable_cause': ['probable cause', 'PC', 'reasonable grounds', 'reasonable belief'],
            'search_warrant': ['warrant', 'search authorization', 'judicial approval', 'magistrate'],
            'automobile_exception': ['automobile exception', 'motor vehicle exception', 'Carroll doctrine'],
            'terry_stop': ['Terry stop', 'investigatory stop', 'reasonable suspicion', 'frisk'],
            'use_of_force': ['use of force', 'deadly force', 'reasonable force', 'excessive force'],
            'arrest': ['arrest', 'custodial arrest', 'arrest warrant', 'warrantless arrest'],
            'oui_dui': ['OUI', 'DUI', 'operating under influence', 'drunk driving'],
            'consent_search': ['consent', 'voluntary consent', 'consent to search'],
            'exigent_circumstances': ['exigent circumstances', 'emergency', 'hot pursuit', 'immediate danger']
        }

    def extract_text_with_structure(self, pdf_path: str) -> List[Dict]:
        """Extract text while preserving document structure"""
        doc = fitz.open(pdf_path)
        structured_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Get text with formatting
            text = page.get_text()
            
            # Get text blocks with position info
            blocks = page.get_text("dict")
            
            structured_content.append({
                'page_number': page_num + 1,
                'text': text,
                'blocks': blocks,
                'has_headings': self._detect_headings(text),
                'structure_score': self._calculate_structure_score(text)
            })
            
        doc.close()
        return structured_content

    def _detect_headings(self, text: str) -> bool:
        """Detect if page contains chapter/section headings"""
        return bool(self.patterns['chapter'].search(text) or 
                   self.patterns['section'].search(text))

    def _calculate_structure_score(self, text: str) -> float:
        """Calculate how well-structured the text appears"""
        score = 0.0
        
        # Check for various structural elements
        if self.patterns['chapter'].search(text):
            score += 0.3
        if self.patterns['section'].search(text):
            score += 0.2
        if self.patterns['mgl_citation'].search(text):
            score += 0.2
        if self.patterns['case_citation'].search(text):
            score += 0.2
        if self.patterns['numbered_list'].search(text):
            score += 0.1
            
        return min(score, 1.0)

    def intelligent_chunking(self, structured_content: List[Dict]) -> List[LegalChunk]:
        """Chunk content while preserving legal structure"""
        chunks = []
        current_chapter = ""
        current_section = ""
        
        for page_data in structured_content:
            text = page_data['text']
            page_num = page_data['page_number']
            
            # Split by strong breaks (chapters, major sections)
            sections = self._split_by_structure(text)
            
            for section_text in sections:
                # Update context
                chapter_match = self.patterns['chapter'].search(section_text)
                if chapter_match:
                    current_chapter = chapter_match.group(1)
                
                section_match = self.patterns['section'].search(section_text)
                if section_match:
                    current_section = section_match.group(1)
                
                # Create chunks that respect legal boundaries
                section_chunks = self._create_semantic_chunks(
                    section_text, 
                    current_chapter, 
                    current_section, 
                    page_num
                )
                chunks.extend(section_chunks)
        
        return chunks

    def _split_by_structure(self, text: str) -> List[str]:
        """Split text at natural legal boundaries"""
        # Split on chapters first
        chapter_splits = re.split(r'\n(?=CHAPTER\s+\d+)', text)
        
        result = []
        for chapter_text in chapter_splits:
            # Further split on major sections
            section_splits = re.split(r'\n(?=Section\s+[\d.]+)', chapter_text)
            result.extend(section_splits)
        
        return [s.strip() for s in result if s.strip()]

    def _create_semantic_chunks(self, text: str, chapter: str, section: str, page_num: int) -> List[LegalChunk]:
        """Create chunks that preserve legal semantics"""
        chunks = []
        
        # Don't split if text is reasonably sized and coherent
        if len(text) < 1500 and self._is_coherent_legal_unit(text):
            chunk = LegalChunk(
                content=text,
                chapter=chapter,
                section=section,
                page_number=page_num,
                citations=self._extract_citations(text),
                legal_concepts=self._extract_legal_concepts(text),
                confidence_score=self._calculate_chunk_quality(text)
            )
            chunks.append(chunk)
            return chunks
        
        # For longer text, split more carefully
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Check if adding this paragraph would break semantic unity
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if (len(test_chunk) > 1000 and 
                self._would_break_semantic_unity(current_chunk, para)):
                
                # Save current chunk
                if current_chunk:
                    chunk = LegalChunk(
                        content=current_chunk,
                        chapter=chapter,
                        section=section,
                        page_number=page_num,
                        citations=self._extract_citations(current_chunk),
                        legal_concepts=self._extract_legal_concepts(current_chunk),
                        confidence_score=self._calculate_chunk_quality(current_chunk)
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk = para
            else:
                current_chunk = test_chunk
        
        # Add final chunk
        if current_chunk:
            chunk = LegalChunk(
                content=current_chunk,
                chapter=chapter,
                section=section,
                page_number=page_num,
                citations=self._extract_citations(current_chunk),
                legal_concepts=self._extract_legal_concepts(current_chunk),
                confidence_score=self._calculate_chunk_quality(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks

    def _is_coherent_legal_unit(self, text: str) -> bool:
        """Check if text represents a coherent legal concept"""
        # Contains definition
        if self.patterns['definition'].search(text):
            return True
        
        # Contains complete elements list
        if (self.patterns['elements_start'].search(text) and 
            len(self.patterns['numbered_list'].findall(text)) >= 2):
            return True
        
        # Single legal concept with citation
        concepts = self._extract_legal_concepts(text)
        citations = self._extract_citations(text)
        
        return len(concepts) <= 2 and len(citations) > 0

    def _would_break_semantic_unity(self, current: str, new_para: str) -> bool:
        """Check if adding new paragraph would break semantic unity"""
        # Don't break in middle of elements list
        if (self.patterns['elements_start'].search(current) and 
            self.patterns['numbered_list'].search(new_para)):
            return False
        
        # Don't break definition from examples
        if ('means' in current or 'defined as' in current) and 'example' in new_para.lower():
            return False
        
        # Break on new major concept
        current_concepts = self._extract_legal_concepts(current)
        new_concepts = self._extract_legal_concepts(new_para)
        
        return len(set(current_concepts) & set(new_concepts)) == 0

    def _extract_citations(self, text: str) -> List[str]:
        """Extract legal citations from text"""
        citations = []
        
        # MGL citations
        mgl_matches = self.patterns['mgl_citation'].findall(text)
        for match in mgl_matches:
            citations.append(f"M.G.L. c. {match[0]}, § {match[1]}")
        
        # Case citations
        case_matches = self.patterns['case_citation'].findall(text)
        for match in case_matches:
            plaintiff, defendant, volume, page, year = match
            citation = f"{plaintiff} v. {defendant}, {volume} Mass. {page}"
            if year:
                citation += f" ({year})"
            citations.append(citation)
        
        return citations

    def _extract_legal_concepts(self, text: str) -> List[str]:
        """Extract legal concepts from text"""
        concepts = []
        text_lower = text.lower()
        
        for concept, keywords in self.legal_concepts.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts

    def _calculate_chunk_quality(self, text: str) -> float:
        """Calculate quality score for chunk"""
        score = 0.0
        
        # Has citations
        if self._extract_citations(text):
            score += 0.3
        
        # Has legal concepts
        if self._extract_legal_concepts(text):
            score += 0.2
        
        # Appropriate length
        length = len(text)
        if 200 <= length <= 1000:
            score += 0.2
        elif 100 <= length <= 1500:
            score += 0.1
        
        # Has structure
        if (self.patterns['numbered_list'].search(text) or 
            self.patterns['definition'].search(text)):
            score += 0.2
        
        # Coherence (not too many different concepts)
        concepts = self._extract_legal_concepts(text)
        if 1 <= len(concepts) <= 3:
            score += 0.1
        
        return min(score, 1.0)

class LegalRAGEvaluator:
    """Evaluation system for legal RAG performance"""
    
    def __init__(self):
        self.test_queries = [
            {
                'query': 'What are the elements of OUI in Massachusetts?',
                'expected_concepts': ['oui_dui'],
                'expected_citations': ['M.G.L. c. 90'],
                'required_terms': ['operate', 'under influence', 'public way'],
                'difficulty': 'basic'
            },
            {
                'query': 'When can police conduct a protective sweep?',
                'expected_concepts': ['search_warrant', 'exigent_circumstances'],
                'expected_citations': ['Maryland v. Buie'],
                'required_terms': ['articulable facts', 'armed', 'dangerous'],
                'difficulty': 'intermediate'
            },
            {
                'query': 'What constitutes a valid Miranda waiver?',
                'expected_concepts': ['miranda_rights'],
                'required_terms': ['knowing', 'intelligent', 'voluntary'],
                'difficulty': 'intermediate'
            },
            {
                'query': 'Explain the automobile exception to warrant requirement',
                'expected_concepts': ['automobile_exception', 'search_warrant', 'probable_cause'],
                'required_terms': ['mobility', 'exigent', 'Carroll'],
                'difficulty': 'advanced'
            },
            {
                'query': 'What are the requirements for a Terry stop?',
                'expected_concepts': ['terry_stop'],
                'required_terms': ['reasonable suspicion', 'articulable facts'],
                'difficulty': 'basic'
            }
        ]

    def evaluate_retrieval_quality(self, rag_system, chunks: List[LegalChunk]) -> Dict:
        """Evaluate retrieval quality"""
        results = {}
        
        for test_case in self.test_queries:
            query = test_case['query']
            
            # Get retrieval results
            retrieved = rag_system.retrieve(query, top_k=5)
            
            # Evaluate relevance
            relevance_score = self._calculate_relevance(retrieved, test_case)
            
            # Check concept coverage
            concept_coverage = self._check_concept_coverage(retrieved, test_case)
            
            # Check citation preservation
            citation_preservation = self._check_citation_preservation(retrieved, test_case)
            
            results[query] = {
                'relevance_score': relevance_score,
                'concept_coverage': concept_coverage,
                'citation_preservation': citation_preservation,
                'retrieved_chunks': len(retrieved),
                'difficulty': test_case['difficulty']
            }
        
        # Calculate aggregate metrics
        avg_relevance = np.mean([r['relevance_score'] for r in results.values()])
        avg_concept_coverage = np.mean([r['concept_coverage'] for r in results.values()])
        avg_citation_preservation = np.mean([r['citation_preservation'] for r in results.values()])
        
        return {
            'individual_results': results,
            'aggregate_metrics': {
                'average_relevance': avg_relevance,
                'average_concept_coverage': avg_concept_coverage,
                'average_citation_preservation': avg_citation_preservation,
                'overall_score': (avg_relevance + avg_concept_coverage + avg_citation_preservation) / 3
            }
        }

    def _calculate_relevance(self, retrieved: List, test_case: Dict) -> float:
        """Calculate relevance score"""
        if not retrieved:
            return 0.0
        
        required_terms = test_case.get('required_terms', [])
        total_relevance = 0.0
        
        for chunk in retrieved:
            chunk_relevance = 0.0
            chunk_text = chunk.content.lower() if hasattr(chunk, 'content') else str(chunk).lower()
            
            # Check for required terms
            for term in required_terms:
                if term.lower() in chunk_text:
                    chunk_relevance += 1.0 / len(required_terms)
            
            total_relevance += chunk_relevance
        
        return min(total_relevance / len(retrieved), 1.0)

    def _check_concept_coverage(self, retrieved: List, test_case: Dict) -> float:
        """Check if expected legal concepts are covered"""
        expected_concepts = test_case.get('expected_concepts', [])
        if not expected_concepts:
            return 1.0
        
        found_concepts = set()
        for chunk in retrieved:
            if hasattr(chunk, 'legal_concepts'):
                found_concepts.update(chunk.legal_concepts)
        
        coverage = len(set(expected_concepts) & found_concepts) / len(expected_concepts)
        return coverage

    def _check_citation_preservation(self, retrieved: List, test_case: Dict) -> float:
        """Check if citations are preserved"""
        expected_citations = test_case.get('expected_citations', [])
        if not expected_citations:
            return 1.0
        
        found_citations = []
        for chunk in retrieved:
            if hasattr(chunk, 'citations'):
                found_citations.extend(chunk.citations)
            else:
                # Check content for citations
                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                # Simple citation detection in text
                if any(cite in chunk_text for cite in expected_citations):
                    found_citations.extend(expected_citations)
        
        if not found_citations:
            return 0.0
        
        # Check overlap
        citation_overlap = 0
        for expected in expected_citations:
            if any(expected in found for found in found_citations):
                citation_overlap += 1
        
        return citation_overlap / len(expected_citations)

def run_comprehensive_evaluation(pdf_path: str) -> Dict:
    """Run complete evaluation of legal RAG system"""
    
    print("🏛️  Massachusetts Criminal Law RAG Evaluation System")
    print("=" * 60)
    
    # Initialize components
    processor = LegalDocumentProcessor()
    evaluator = LegalRAGEvaluator()
    
    print("📖 Extracting and processing legal document...")
    
    # Extract and process document
    structured_content = processor.extract_text_with_structure(pdf_path)
    print(f"   Extracted {len(structured_content)} pages")
    
    # Create intelligent chunks
    chunks = processor.intelligent_chunking(structured_content)
    print(f"   Created {len(chunks)} intelligent legal chunks")
    
    # Analyze chunk quality
    quality_scores = [chunk.confidence_score for chunk in chunks]
    avg_quality = np.mean(quality_scores)
    
    print(f"   Average chunk quality: {avg_quality:.3f}")
    print(f"   Chunks with citations: {sum(1 for c in chunks if c.citations)}")
    print(f"   Chunks with legal concepts: {sum(1 for c in chunks if c.legal_concepts)}")
    
    # Analyze structure preservation
    chapters = set(c.chapter for c in chunks if c.chapter)
    sections = set(c.section for c in chunks if c.section)
    
    print(f"   Identified chapters: {len(chapters)}")
    print(f"   Identified sections: {len(sections)}")
    
    # Sample chunk analysis
    print("\n📋 Sample Chunk Analysis:")
    high_quality_chunks = sorted(chunks, key=lambda x: x.confidence_score, reverse=True)[:3]
    
    for i, chunk in enumerate(high_quality_chunks, 1):
        print(f"\n   Chunk {i} (Quality: {chunk.confidence_score:.3f}):")
        print(f"   Chapter: {chunk.chapter}, Section: {chunk.section}")
        print(f"   Citations: {chunk.citations}")
        print(f"   Concepts: {chunk.legal_concepts}")
        print(f"   Content preview: {chunk.content[:150]}...")
    
    # TODO: If you have an existing RAG system, uncomment this:
    # print("\n🔍 Evaluating Retrieval Performance...")
    # rag_evaluation = evaluator.evaluate_retrieval_quality(your_rag_system, chunks)
    # print(f"   Overall Score: {rag_evaluation['aggregate_metrics']['overall_score']:.3f}")
    
    print("\n✅ Evaluation Complete!")
    print("\nRecommendations:")
    
    if avg_quality < 0.6:
        print("⚠️  Chunk quality below target - consider improving structure detection")
    
    citation_coverage = sum(1 for c in chunks if c.citations) / len(chunks)
    if citation_coverage < 0.3:
        print("⚠️  Low citation coverage - improve citation extraction patterns")
    
    concept_coverage = sum(1 for c in chunks if c.legal_concepts) / len(chunks)
    if concept_coverage < 0.5:
        print("⚠️  Low concept coverage - expand legal concept dictionary")
    
    return {
        'total_chunks': len(chunks),
        'average_quality': avg_quality,
        'citation_coverage': citation_coverage,
        'concept_coverage': concept_coverage,
        'chapters_identified': len(chapters),
        'sections_identified': len(sections),
        'chunks': chunks
    }

if __name__ == "__main__":
    # Example usage
    pdf_path = "path/to/scheft_criminal_law_2025.pdf"
    
    try:
        results = run_comprehensive_evaluation(pdf_path)
        
        # Save results for further analysis
        with open('legal_rag_evaluation_results.json', 'w') as f:
            # Convert chunks to serializable format
            serializable_results = {k: v for k, v in results.items() if k != 'chunks'}
            json.dump(serializable_results, f, indent=2)
            
        print(f"\n💾 Results saved to legal_rag_evaluation_results.json")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()