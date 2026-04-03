# Chapter 2 - Quick Practice

Minimal scripts for concept practice (no complex architecture).

## Chapter 2 (`chapter02_run.py`)

Run all demos:

```cmd
run_chapter02.bat
```

Run only one section:

```cmd
run_chapter02.bat --mode tokens
run_chapter02.bat --mode embeddings --device cuda
run_chapter02.bat --mode generation
```

## Chapter 2.5 RAG (`chapter02_5_rag_run.py`)

Run basic RAG example:

```cmd
run_chapter02_5_rag.bat
```

Custom query:

```cmd
run_chapter02_5_rag.bat --query "Como se convierten frases en vectores?" --n-results 2
```

If cache path gives permissions error:

```cmd
run_chapter02_5_rag.bat --cache-home .chroma_home_user
```
