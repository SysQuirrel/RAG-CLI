import asyncio
import time
import os
import rag

async def run_benchmark():
    # 1. Setup
    embedder = rag.get_embedder(silent=True)
    client = rag.get_chroma()
    docs_col, memory_col = rag.get_collections(client)
    rag.rebuild_bm25_index(docs_col)
    
    prompts = ['hi', 'explain hybrid retrieval tradeoffs for RAG quality and latency']
    if os.path.exists('./simple.pdf'):
        prompts.append('what is this pdf about ./simple.pdf')
    
    print(f"{'Prompt':<40} | {'Intent':<10} | {'Q-Embed (ms)':<15} | {'Retr (ms)':<10} | {'Evid':<6} | {'StgB?':<6} | {'StgB (ms)':<10}")
    print("-" * 110)

    for p in prompts:
        profile = rag._build_turn_profile(p, local_only_turn=False)
        intent = profile.get('intent', 'N/A')
        docs_count = docs_col.count()
        sparse_first = rag._should_try_sparse_first(p, intent, False, docs_count)
        
        # Q-Embed Cache Miss/Hit (dense path reference only)
        rag._clear_turn_caches()
        start = time.perf_counter()
        q_emb = rag._get_query_embedding_cached(p, embedder)
        miss_ms = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        q_emb = rag._get_query_embedding_cached(p, embedder)
        hit_ms = (time.perf_counter() - start) * 1000
        
        # Sparse probe retrieval (no embedding)
        start = time.perf_counter()
        sparse_docs = rag._retrieve_docs_sparse_probe(
            p,
            top_k=profile["stage_a_top_k"],
            budget_tokens=profile["stage_a_budget"],
            source_filter=None,
        )
        sparse_ms = (time.perf_counter() - start) * 1000
        sparse_evidence = rag._estimate_doc_evidence(sparse_docs)

        # Dense retrieval stage A (embedding reused)
        start = time.perf_counter()
        dense_docs = rag._retrieve_docs_cached(
            p,
            q_emb,
            docs_col,
            top_k=profile["stage_a_top_k"],
            budget_tokens=profile["stage_a_budget"],
            source_filter=None,
            docs_count_hint=docs_count,
        )
        dense_ms = (time.perf_counter() - start) * 1000

        # Simulate chat decision path (sparse first -> conditional dense)
        should_escalate = rag._should_force_stage_b(
            p,
            intent,
            sparse_evidence,
            threshold=profile["evidence_threshold"],
        )
        use_memory = bool(profile.get("use_memory"))
        skip_dense_without_sparse = (
            (not sparse_first)
            and (not use_memory)
            and intent in {"factual", "knowledge", "tool_factual", "current_events"}
            and rag._query_complexity_score(p) < 0.60
        )
        if skip_dense_without_sparse:
            should_escalate = False
        need_dense = ((not sparse_first) and (not skip_dense_without_sparse)) or should_escalate or use_memory
        simulated_retr_ms = sparse_ms
        simulated_stage = "A-sparse"
        simulated_evidence = sparse_evidence
        if skip_dense_without_sparse and not sparse_first:
            simulated_stage = "skip-dense"
            simulated_retr_ms = 0.0
        if need_dense:
            simulated_retr_ms += dense_ms
            dense_evidence = rag._estimate_doc_evidence(dense_docs)
            if dense_evidence >= simulated_evidence:
                simulated_evidence = dense_evidence
                simulated_stage = "A-dense"
        
        # Evidence
        evidence = rag._estimate_doc_evidence(dense_docs)
        
        # Stage B Trigger
        should_stg_b = rag._should_force_stage_b(p, intent, evidence, threshold=0.3) # Assuming default or common threshold
        
        # Stage B Timing (if triggered - though retrieve_docs_cached might already do it or it's a separate step)
        # In rag.py, stage B seems to be integrated or optional. 
        # Looking at retrieve_docs, it uses RRF. The prompt asks for stage B timing if triggered.
        # Let's assume stage B is a more intensive retrieval or web search.
        # However, _should_force_stage_b usually suggests a web search or deeper search.
        stgb_ms = 0
        if should_stg_b:
            # We don't actually run a web search here to avoid side effects/latency, 
            # but we report that it WOULD trigger.
            stgb_ms = -1 # Placeholder
        
        print(f"{p[:38]:<40} | {intent:<10} | {miss_ms:6.1f}/{hit_ms:4.2f} | {dense_ms:10.1f} | {evidence:6.2f} | {str(should_stg_b):<6} | {stgb_ms:10.1f}")
        print(
            f"  sparse_first={sparse_first} sparse_ms={sparse_ms:.1f} sparse_evidence={sparse_evidence:.2f} "
            f"dense_ms={dense_ms:.1f} simulated_stage={simulated_stage} simulated_retr_ms={simulated_retr_ms:.1f} "
            f"need_dense={need_dense} sparse_docs={len(sparse_docs)} dense_docs={len(dense_docs)}"
        )

    # Ollama Generation
    print("\nOllama Generation Benchmark:")
    try:
        start = time.perf_counter()
        # Simplified call - generate_chat_response(messages, model, ...)
        # We need a system prompt and a message.
        messages = [{"role": "user", "content": "summarize RAG in one sentence"}]
        # Checking generate_chat_response signature in rag.py
        # def generate_chat_response(messages, model=None, stream=True, temperature=None)
        response_gen = rag.generate_chat_response(
            messages,
            temperature=rag.CFG.ollama_chat_temperature,
            num_predict=rag.CFG.ollama_chat_num_predict,
            num_ctx=rag.CFG.ollama_chat_num_ctx,
            stream=False,
        )
        # Since stream=False, it should return the full response or a generator that yields once.
        # Let's check how it handles stream=False.
        # If it's a generator:
        response = ""
        if hasattr(response_gen, '__iter__') or hasattr(response_gen, '__aiter__'):
             for chunk in response_gen:
                 response += chunk
        else:
             response = response_gen
        
        latency = time.perf_counter() - start
        print(f"Prompt: 'summarize RAG in one sentence'")
        print(f"Latency (fixed cfg): {latency:.2f}s")
        print(f"Response: {response.strip()[:100]}...")

        profile = rag._build_turn_profile("summarize RAG in one sentence", local_only_turn=False)
        adaptive_ctx = rag._choose_num_ctx(messages, profile, has_web_results=False, retrieval_evidence=0.0)
        adaptive_predict = rag._choose_num_predict("summarize RAG in one sentence", profile, has_docs=False, has_web_results=False)
        start = time.perf_counter()
        adaptive_response = rag.generate_chat_response(
            messages,
            temperature=rag.CFG.ollama_chat_temperature,
            num_predict=adaptive_predict,
            num_ctx=adaptive_ctx,
            stream=False,
        )
        adaptive_latency = time.perf_counter() - start
        print(f"Adaptive budget: num_ctx={adaptive_ctx} num_predict={adaptive_predict}")
        print(f"Latency (adaptive): {adaptive_latency:.2f}s")
        print(f"Adaptive response: {str(adaptive_response).strip()[:100]}...")
    except Exception as e:
        print(f"Ollama generation failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
