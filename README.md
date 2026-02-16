# CommunityDocs RAG

A review-grounded question-answering system that explains what people have said about a café, with citations.

## Problem

Neurodivergent people and others with sensory sensitivities often struggle to choose cafés and restaurants where they can comfortably converse or focus. Existing platforms (e.g., Google Maps) contain relevant information in reviews, but it is **unstructured, inconsistent, and difficult to interpret**.

## Solution

**CommunityDocs RAG** lets users ask questions about a café and receive **evidence-backed answers grounded in actual review text**. Every claim is cited, and uncertainty is made explicit.

### Example
- **User asks:** "Is it loud in the evening?"
- **System responds:** "Multiple reviews mention noise in the evening (3/12 reviews). One reviewer noted: *'Gets pretty rowdy after 7pm on weekends.'* However, some recent reviews suggest it's quieter mid-week."

## Core Features

### Feature 1: Ask Reviews (ReviewRAG)
For each venue:
- **Natural-language Q&A** - Ask anything about the café
- **Grounded answers** - Every claim backed by actual review text
- **Citations** - See the exact review excerpts supporting each answer
- **Honest uncertainty** - Explicit handling of insufficient evidence

### Feature 2: Sound Profile (SoundSense)
*Companion project TBD in `soundsense-nlp`*

For each venue:
- Natural-language summary of acoustic characteristics
- Time-of-day sound levels (when supported)
- Confidence scores
- Key contributing factors
- Clear communication of limitations

## Technical Approach

### CommunityDocs RAG
- **Embedding-based retrieval** - Find relevant review chunks using vector search
- **LLM answer generation** - Generate natural-language answers with strict citation enforcement
- **No hallucinations** - System refuses to make claims without evidence

### Data Sources
- Public review datasets (e.g., Yelp Open Dataset)
- Synthetic or sampled subsets for MVP

## Target Users

- Neurodivergent individuals with sound sensitivity
- People seeking calm meeting spaces
- Accessibility advocates
- Researchers and NGOs exploring sensory accessibility

## What This Is (And Isn't)

### What This Project Does
- Provides evidence-backed explanations, not opaque scores
- Surfaces uncertainty honestly
- Surface patterns and explain them transparently

### What This Project Doesn't Do
- Real-time sound measurement
- Google Maps integration
- Definitive or authoritative ratings
- City-wide completeness claims
- Present outputs as objective truth

**This is an assistive decision-support MVP.**

## Installation & Setup

*Coming soon - project under development*

```bash
# Clone the repo
git clone <repo-url>
cd communitydocs-rag

# Set up environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

## Project Structure

```
communitydocs-rag/
├── src/communitydocs_rag/
│   ├── config.py           # Configuration
│   ├── ingestion/          # Data loading & preprocessing
│   ├── retrieval/          # Vector search & retrieval
│   ├── llm/                # LLM interactions
│   ├── api/                # API endpoints
│   └── eval/               # Evaluation scripts
├── tests/                  # Test suite
└── pyproject.toml          # Project metadata & dependencies
```

## Related Project

**SoundSense** (`soundsense-nlp`)  
A weakly supervised NLP system that infers acoustic accessibility signals from reviews, with uncertainty.  

## Success Metrics (MVP)

- Retrieval relevance (manual spot checks)
- Citation correctness
- Schema validity rate
- Qualitative plausibility of answers

## Future Extensions (Out of Scope)

- User-submitted ratings
- Real-time sensing
- Map-based UI
- Multilingual expansion

## License

CC-BY-NC-4.0 - This work is licensed for non-commercial use.

## Authors

- chrissab-dev

---

**Status:** Early development/MVP phase
