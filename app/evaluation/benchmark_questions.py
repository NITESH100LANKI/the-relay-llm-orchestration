"""
15 Self-Authored Evaluation Questions for The Relay.
Carefully balanced across:
- Direct factual questions
- Deep technical questions
- Cross-section questions
- Cross-paper comparative questions
- Questions with incomplete evidence
- Questions not answered by any of the three papers (unanswerable)
- Adversarial questions likely to mislead a naive retriever (lexical traps, cross-paper collisions)
"""

from typing import List, Dict, Any

BENCHMARK_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "Q01",
        "category": "direct_factual",
        "question": "What is the exact mathematical formula for Scaled Dot-Product Attention in Attention Is All You Need?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "expected_page": 4,
        "keywords": ["Softmax", "QK^T", "sqrt(d_k)", "V"]
    },
    {
        "id": "Q02",
        "category": "direct_factual",
        "question": "What is the bitrate and frame rate of the Mimi audio codec in Moshi?",
        "expected_grade": "correct",
        "target_paper": "Moshi: a speech-text foundation model for real-time dialogue",
        "expected_page": 2,
        "keywords": ["1.1 kbps", "12.5 Hz"]
    },
    {
        "id": "Q03",
        "category": "direct_factual",
        "question": "How does Dynamic Chunking determine chunk boundaries for hierarchical sequence modeling?",
        "expected_grade": "correct",
        "target_paper": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "expected_page": 4,
        "keywords": ["entropy", "boundary", "dynamic"]
    },
    {
        "id": "Q04",
        "category": "technical",
        "question": "In the base Transformer model, what are the values of d_model, d_k, d_v, and the number of heads h?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "expected_page": 4,
        "keywords": ["d_model = 512", "h = 8", "d_k = 64"]
    },
    {
        "id": "Q05",
        "category": "technical",
        "question": "How many codebooks does Mimi use in its Residual Vector Quantizer (RVQ) and what is the codebook size?",
        "expected_grade": "correct",
        "target_paper": "Moshi: a speech-text foundation model for real-time dialogue",
        "expected_page": 4,
        "keywords": ["codebooks", "RVQ", "quantizer"]
    },
    {
        "id": "Q06",
        "category": "cross_section",
        "question": "How does positional encoding in Section 3.5 of Attention Is All You Need address sequence order without recurrence?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "expected_page": 5,
        "keywords": ["sinusoid", "frequencies", "positional encoding"]
    },
    {
        "id": "Q07",
        "category": "cross_section",
        "question": "In Moshi, how are text tokens and audio codec tokens synchronized across the multi-stream channels?",
        "expected_grade": "correct",
        "target_paper": "Moshi: a speech-text foundation model for real-time dialogue",
        "expected_page": 3,
        "keywords": ["multi-stream", "synchronized", "delay", "interleaving"]
    },
    {
        "id": "Q08",
        "category": "cross_paper_comparison",
        "question": "How does sequence downsampling in Dynamic Chunking compare conceptually to the multi-head attention mechanism in Attention Is All You Need?",
        "expected_grade": "correct",
        "target_paper": "Both",
        "expected_page": 4,
        "keywords": ["downsampling", "attention", "hierarchical"]
    },
    {
        "id": "Q09",
        "category": "cross_paper_comparison",
        "question": "Compare the latency handling in Moshi's streaming architecture with the auto-regressive generation in the original Transformer.",
        "expected_grade": "correct",
        "target_paper": "Both",
        "expected_page": 2,
        "keywords": ["latency", "streaming", "autoregressive"]
    },
    {
        "id": "Q10",
        "category": "incomplete_evidence",
        "question": "What was the exact room temperature in degrees Celsius of the Google server room where the Transformer was trained?",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "expected_page": None,
        "keywords": ["insufficient evidence", "not mentioned"]
    },
    {
        "id": "Q11",
        "category": "unanswerable",
        "question": "What is the exact financial dollar cost of electricity and GPU hardware used to train the Moshi 7B model?",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "expected_page": None,
        "keywords": ["insufficient evidence", "not mentioned"]
    },
    {
        "id": "Q12",
        "category": "unanswerable",
        "question": "How does the Transformer model perform when quantized and deployed on an Apple Watch Series 9?",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "expected_page": None,
        "keywords": ["insufficient evidence", "not mentioned"]
    },
    {
        "id": "Q13",
        "category": "naive_retriever_trap",
        "question": "What is the role of the acoustic Transformer in dynamic chunking sequence modeling?",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "expected_page": None,
        "keywords": ["insufficient evidence", "does not describe an acoustic Transformer"]
    },
    {
        "id": "Q14",
        "category": "naive_retriever_trap",
        "question": "How does Mimi's audio codec utilize 8 parallel multi-head attention layers in its residual vector quantizer?",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "expected_page": None,
        "keywords": ["insufficient evidence", "convolutional", "not multi-head"]
    },
    {
        "id": "Q15",
        "category": "naive_retriever_trap",
        "question": "Why did Vaswani et al. choose a 12.5 Hz latent frame rate for the Transformer encoder self-attention?",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "expected_page": None,
        "keywords": ["insufficient evidence", "does not use 12.5 Hz", "12.5 Hz belongs to Moshi"]
    }
]

