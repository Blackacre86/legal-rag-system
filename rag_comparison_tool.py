#!/usr/bin/env python3
"""
RAG System Comparison & Optimization Tool
Compare your current basic RAG against the optimized legal version
"""

import json
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# Imports for your existing system
import fitz  # PyMuPDF  
import faiss
from sentence_transformers import SentenceTransformer

@dataclass
class ComparisonResult:
    """Results from comparing two RAG systems"""
    test_query: str
    
    # Basic system results
    basic_response_time: float
    basic_answer: str
    basic_sources_count: int
    basic_chunk_quality: float
    
    # Optimized system results  
    optimized_response_time: float
    optimized_answer: str
    optimized_sources_count: int
    optimized_chunk_quality: float
    
    # Comparison metrics
    accuracy_improvement: float
    speed_improvement: float
    quality_improvement: float

class BasicRAGSystem:
    """Wrapper for your existing basic RAG system"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Process document using your existing approach
        self._process_document()
        
    def _process_document(self):
        """Your existing basic processing"""
        doc = fitz.open(self.pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        
        # Basic chunking (your current approach)
        paragraphs = text.split('\n\n')
        self.chunks = [p.strip() for p in paragraphs if len(p.strip()) > 100]
        
        print(f"Basic system: Created {len(self.chunks)} chunks")
        
        # Create embeddings
        self.embeddings = self.model.encode(self.chunks)
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype('float32'))
    
    def query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Your existing query method"""
        start_time = time.time()
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({
                    'content': self.chunks[idx],
                    'distance': float(dist),
                    'similarity': 1.0 / (1.0 + dist)  # Convert distance to similarity
                })
        
        response_time = time.time() - start_time
        
        # Simple answer generation (concatenate top results)
        answer = "Based on the retrieved information:\n\n"
        for i, result in enumerate(results[:2], 1):
            answer += f"{i}. {result['content'][:200]}...\n\n"
        
        return {
            'answer': answer,
            'sources': results,
            'response_time': response_time,
            'source_count': len(results)
        }

class RAGSystemComparator:
    """Compare basic vs optimized RAG systems"""
    
    def __init__(self):
        self.test_queries = [
            "What are the elements of OUI in Massachusetts?",
            "When can police conduct a protective sweep?", 
            "What constitutes a valid Miranda waiver?",
            "Explain the automobile exception to warrant requirement",
            "What are the requirements for a Terry stop?",
            "How does the exclusionary rule apply?",
            "When can police enter a home without a warrant?",
            "What is the penalty for first offense OUI?"
        ]
        
    def compare_systems(self, basic_system, optimized_system) -> List[ComparisonResult]:
        """Run comprehensive comparison"""
        
        print("🔍 Running RAG System Comparison")
        print("="*50)
        
        results = []
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"\nTest {i}/{len(self.test_queries)}: {query[:50]}...")
            
            # Test basic system
            print("   Testing basic system...")
            basic_result = basic_system.query(query)
            
            # Test optimized system
            print("   Testing optimized system...")
            optimized_result = optimized_system.query(query)
            
            # Calculate quality scores
            basic_quality = self._assess_response_quality(basic_result['answer'], query)
            optimized_quality = self._assess_response_quality(optimized_result['answer'], query)
            
            # Create comparison result
            comparison = ComparisonResult(
                test_query=query,
                basic_response_time=basic_result['response_time'],
                basic_answer=basic_result['answer'],
                basic_sources_count=basic_result['source_count'],
                basic_chunk_quality=basic_quality,
                optimized_response_time=optimized_result.get('response_time', 0),
                optimized_answer=optimized_result.get('answer', ''),
                optimized_sources_count=len(optimized_result.get('sources', [])),
                optimized_chunk_quality=optimized_quality,
                accuracy_improvement=(optimized_quality - basic_quality) / basic_quality if basic_quality > 0 else 0,
                speed_improvement=(basic_result['response_time'] - optimized_result.get('response_time', 0)) / basic_result['response_time'] if basic_result['response_time'] > 0 else 0,
                quality_improvement=(optimized_quality - basic_quality)
            )
            
            results.append(comparison)
            
            print(f"   Basic quality: {basic_quality:.3f}")
            print(f"   Optimized quality: {optimized_quality:.3f}")
            print(f"   Improvement: {comparison.quality_improvement:.3f}")
        
        return results
    
    def _assess_response_quality(self, answer: str, query: str) -> float:
        """Simple quality assessment"""
        score = 0.0
        answer_lower = answer.lower()
        
        # Check for legal citations
        if 'm.g.l.' in answer_lower or 'mass.' in answer_lower:
            score += 0.3
        
        # Check for case references
        if ' v. ' in answer:
            score += 0.2
        
        # Check for legal terminology
        legal_terms = ['elements', 'requirements', 'probable cause', 'reasonable suspicion', 
                      'warrant', 'miranda', 'fourth amendment']
        found_terms = sum(1 for term in legal_terms if term in answer_lower)
        score += min(found_terms * 0.05, 0.3)
        
        # Check length (not too short, not too long)
        if 200 <= len(answer) <= 1000:
            score += 0.2
        elif 100 <= len(answer) <= 1500:
            score += 0.1
        
        return min(score, 1.0)
    
    def generate_comparison_report(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        
        # Calculate aggregate metrics
        avg_basic_quality = np.mean([r.basic_chunk_quality for r in results])
        avg_optimized_quality = np.mean([r.optimized_chunk_quality for r in results])
        avg_quality_improvement = np.mean([r.quality_improvement for r in results])
        
        avg_basic_time = np.mean([r.basic_response_time for r in results])
        avg_optimized_time = np.mean([r.optimized_response_time for r in results])
        avg_speed_improvement = np.mean([r.speed_improvement for r in results])
        
        avg_basic_sources = np.mean([r.basic_sources_count for r in results])
        avg_optimized_sources = np.mean([r.optimized_sources_count for r in results])
        
        report = {
            "summary": {
                "total_tests": len(results),
                "average_quality_improvement": avg_quality_improvement,
                "average_speed_improvement": avg_speed_improvement,
                "basic_system_quality": avg_basic_quality,
                "optimized_system_quality": avg_optimized_quality,
                "basic_system_speed": avg_basic_time,
                "optimized_system_speed": avg_optimized_time,
                "basic_sources_per_query": avg_basic_sources,
                "optimized_sources_per_query": avg_optimized_sources
            },
            "detailed_results": [
                {
                    "query": r.test_query,
                    "quality_improvement": r.quality_improvement,
                    "speed_improvement": r.speed_improvement,
                    "basic_quality": r.basic_chunk_quality,
                    "optimized_quality": r.optimized_chunk_quality
                }
                for r in results
            ],
            "recommendations": self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[ComparisonResult]) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        # Quality analysis
        avg_quality_improvement = np.mean([r.quality_improvement for r in results])
        if avg_quality_improvement > 0.3:
            recommendations.append("✅ Significant quality improvement detected - proceed with optimized system")
        elif avg_quality_improvement > 0.1:
            recommendations.append("📈 Moderate quality improvement - consider adopting optimized approaches")
        else:
            recommendations.append("⚠️ Limited quality improvement - investigate specific optimizations")
        
        # Speed analysis
        avg_speed_improvement = np.mean([r.speed_improvement for r in results])
        if avg_speed_improvement > 0:
            recommendations.append("⚡ Performance improvement achieved")
        else:
            recommendations.append("🐌 Performance regression detected - optimize indexing and retrieval")
        
        # Source quality analysis
        poor_performers = [r for r in results if r.basic_chunk_quality < 0.5]
        if len(poor_performers) > len(results) * 0.3:
            recommendations.append("🔧 High number of poor-quality responses - implement legal-aware chunking")
        
        # Specific improvements
        no_citations = [r for r in results if 'm.g.l.' not in r.basic_answer.lower() and 'mass.' not in r.basic_answer.lower()]
        if len(no_citations) > len(results) * 0.5:
            recommendations.append("📚 Many responses lack legal citations - improve citation extraction")
        
        return recommendations
    
    def visualize_comparison(self, results: List[ComparisonResult], save_path: str = None):
        """Create visualization of comparison results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RAG System Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Quality comparison
        queries = [r.test_query[:30] + '...' for r in results]
        basic_quality = [r.basic_chunk_quality for r in results]
        optimized_quality = [r.optimized_chunk_quality for r in results]
        
        x = np.arange(len(queries))
        width = 0.35
        
        axes[0,0].bar(x - width/2, basic_quality, width, label='Basic System', alpha=0.7)
        axes[0,0].bar(x + width/2, optimized_quality, width, label='Optimized System', alpha=0.7)
        axes[0,0].set_xlabel('Test Queries')
        axes[0,0].set_ylabel('Quality Score')
        axes[0,0].set_title('Response Quality Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(queries, rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Response time comparison
        basic_time = [r.basic_response_time for r in results]
        optimized_time = [r.optimized_response_time for r in results]
        
        axes[0,1].bar(x - width/2, basic_time, width, label='Basic System', alpha=0.7)
        axes[0,1].bar(x + width/2, optimized_time, width, label='Optimized System', alpha=0.7)
        axes[0,1].set_xlabel('Test Queries')
        axes[0,1].set_ylabel('Response Time (seconds)')
        axes[0,1].set_title('Response Time Comparison')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(queries, rotation=45, ha='right')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Quality improvement distribution
        quality_improvements = [r.quality_improvement for r in results]
        axes[1,0].hist(quality_improvements, bins=10, alpha=0.7, edgecolor='black')
        axes[1,0].axvline(np.mean(quality_improvements), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(quality_improvements):.3f}')
        axes[1,0].set_xlabel('Quality Improvement')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Quality Improvements')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Sources retrieved comparison
        basic_sources = [r.basic_sources_count for r in results]
        optimized_sources = [r.optimized_sources_count for r in results]
        
        axes[1,1].scatter(basic_sources, optimized_sources, alpha=0.7, s=100)
        axes[1,1].plot([0, max(max(basic_sources), max(optimized_sources))], 
                       [0, max(max(basic_sources), max(optimized_sources))], 
                       'r--', alpha=0.5, label='Equal Performance')
        axes[1,1].set_xlabel('Basic System Sources')
        axes[1,1].set_ylabel('Optimized System Sources')
        axes[1,1].set_title('Sources Retrieved Comparison')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def print_comparison_summary(self, results: List[ComparisonResult]):
        """Print detailed comparison summary"""
        
        report = self.generate_comparison_report(results)
        
        print("\n" + "="*80)
        print("🔍 RAG SYSTEM COMPARISON SUMMARY")
        print("="*80)
        
        summary = report['summary']
        
        print(f"\n📊 AGGREGATE RESULTS")
        print(f"   Total Test Queries:        {summary['total_tests']}")
        print(f"   Average Quality (Basic):   {summary['basic_system_quality']:.3f}")
        print(f"   Average Quality (Optimized): {summary['optimized_system_quality']:.3f}")
        print(f"   Quality Improvement:       {summary['average_quality_improvement']:.3f} ({summary['average_quality_improvement']*100:+.1f}%)")
        print(f"   Average Speed (Basic):     {summary['basic_system_speed']:.3f}s")
        print(f"   Average Speed (Optimized): {summary['optimized_system_speed']:.3f}s")
        print(f"   Speed Improvement:         {summary['average_speed_improvement']:.3f} ({summary['average_speed_improvement']*100:+.1f}%)")
        
        print(f"\n📈 DETAILED PERFORMANCE")
        for result in results:
            print(f"   {result.test_query[:50]:50} | Quality: {result.quality_improvement:+.3f} | Speed: {result.speed_improvement:+.3f}")
        
        print(f"\n💡 RECOMMENDATIONS")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        # Verdict
        avg_quality_improvement = summary['average_quality_improvement']
        if avg_quality_improvement > 0.3:
            print(f"\n🎯 VERDICT: STRONG IMPROVEMENT - Implement optimized system immediately")
        elif avg_quality_improvement > 0.1:
            print(f"\n🎯 VERDICT: MODERATE IMPROVEMENT - Consider selective optimizations")
        else:
            print(f"\n🎯 VERDICT: MINIMAL IMPROVEMENT - Focus on specific problem areas")
        
        print("="*80)

def run_system_comparison(pdf_path: str, optimized_rag_system=None):
    """Run complete comparison between basic and optimized systems"""
    
    print("🚀 Initializing RAG System Comparison")
    print("="*50)
    
    # Initialize basic system
    print("📚 Setting up basic RAG system...")
    basic_system = BasicRAGSystem(pdf_path)
    
    # Initialize optimized system (placeholder)
    if optimized_rag_system is None:
        print("⚠️  No optimized system provided - using mock results")
        # For demo purposes, create a mock optimized system
        optimized_rag_system = MockOptimizedSystem()
    
    # Run comparison
    comparator = RAGSystemComparator()
    results = comparator.compare_systems(basic_system, optimized_rag_system)
    
    # Generate report
    comparator.print_comparison_summary(results)
    
    # Create visualization
    comparator.visualize_comparison(results, 'rag_comparison_results.png')
    
    # Save detailed results
    report = comparator.generate_comparison_report(results)
    with open('rag_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to rag_comparison_report.json")
    
    return results, report

class MockOptimizedSystem:
    """Mock optimized system for demonstration"""
    
    def query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Mock optimized query with better results"""
        
        # Simulate improved performance
        time.sleep(0.1)  # Faster response time
        
        # Mock better answer with legal citations
        answer = f"""Based on Massachusetts criminal law sources:

The query "{query}" involves several key legal elements. Under M.G.L. Chapter 90, Section 24, the requirements include specific statutory elements that must be proven beyond a reasonable doubt.

Key case law includes Commonwealth v. Smith, 450 Mass. 123 (2010), which established the precedent for this area of law.

The essential elements are:
1. Intent requirement
2. Action requirement  
3. Jurisdictional requirement

See also relevant constitutional protections under the Fourth Amendment and Miranda v. Arizona, 384 U.S. 436 (1966).
"""
        
        return {
            'answer': answer,
            'sources': [
                {'source_file': 'scheft_criminal_law_2025.pdf', 'chapter': '3', 'section': '1'},
                {'source_file': 'scheft_criminal_procedure_2025.pdf', 'chapter': '2', 'section': '4'}
            ],
            'response_time': 0.1,
            'legal_citations': ['M.G.L. c. 90, § 24', 'Commonwealth v. Smith'],
            'query_id': 'mock_123'
        }

if __name__ == "__main__":
    print("🔍 RAG System Comparison Tool")
    print("   Ready to compare basic vs optimized RAG systems")
    print("\nUsage:")
    print("   results, report = run_system_comparison('path/to/scheft_manual.pdf')")