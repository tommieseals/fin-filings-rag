"""Evaluation harness for the RAG system."""
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag import RAGEngine


def load_testset(path: str = "eval/testset.jsonl") -> list[dict]:
    """Load test questions from JSONL file."""
    tests = []
    with open(path) as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line))
    return tests


def evaluate(testset_path: str = "eval/testset.jsonl") -> dict:
    """Run evaluation on the test set.
    
    Args:
        testset_path: Path to the test set JSONL file
        
    Returns:
        Evaluation metrics
    """
    engine = RAGEngine()
    
    if not engine.is_ready():
        print("ERROR: Index not ready. Run ingestion first.")
        return {"error": "index_not_ready"}
    
    tests = load_testset(testset_path)
    
    results = {
        "total": len(tests),
        "answered": 0,
        "abstained": 0,
        "correct": 0,
        "citations_provided": 0,
        "avg_confidence": 0.0,
        "details": [],
    }
    
    confidence_sum = 0.0
    
    for test in tests:
        question = test["question"]
        expected_abstain = test.get("expect_abstain", False)
        expected_keywords = test.get("expected_keywords", [])
        
        response = engine.ask(question)
        
        detail = {
            "question": question,
            "answer": response.answer,
            "abstained": response.abstained,
            "confidence": response.confidence,
            "citations": len(response.citations),
        }
        
        if response.abstained:
            results["abstained"] += 1
            detail["correct"] = expected_abstain
        else:
            results["answered"] += 1
            # Check if expected keywords appear in answer
            if expected_keywords:
                found = any(kw.lower() in response.answer.lower() for kw in expected_keywords)
                detail["correct"] = found
            else:
                detail["correct"] = not expected_abstain
        
        if detail["correct"]:
            results["correct"] += 1
        
        if response.citations:
            results["citations_provided"] += 1
        
        confidence_sum += response.confidence
        results["details"].append(detail)
    
    results["avg_confidence"] = confidence_sum / len(tests) if tests else 0.0
    results["accuracy"] = results["correct"] / len(tests) if tests else 0.0
    
    return results


def print_report(results: dict) -> None:
    """Print evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Total questions:    {results['total']}")
    print(f"Answered:          {results['answered']}")
    print(f"Abstained:         {results['abstained']}")
    print(f"Correct:           {results['correct']}")
    print(f"Accuracy:          {results['accuracy']:.1%}")
    print(f"Avg confidence:    {results['avg_confidence']:.3f}")
    print(f"With citations:    {results['citations_provided']}")
    
    print("\nDetails:")
    print("-" * 60)
    for d in results["details"]:
        status = "✓" if d["correct"] else "✗"
        print(f"{status} Q: {d['question'][:50]}...")
        print(f"  → {'ABSTAINED' if d['abstained'] else d['answer'][:60]}...")
        print(f"  Confidence: {d['confidence']:.3f}, Citations: {d['citations']}")
        print()


if __name__ == "__main__":
    results = evaluate()
    print_report(results)
    
    # Exit with error if accuracy too low
    if results.get("accuracy", 0) < 0.5:
        sys.exit(1)
