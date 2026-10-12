import asyncio
import asyncpg
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.config import get_rag_settings

async def main():
    settings = get_rag_settings()
    dsn = settings.rag_database_url or settings.database_url
    
    conn = await asyncpg.connect(dsn)
    try:
        doc_id = "doc_f6c67c8cb5fd4fd99980d89b3609347b"
        query_text = "What is Maaz’s CGPA?"
        
        # Test term-by-term weighted score sum
        sql = """
            WITH query_lexemes AS (
                SELECT DISTINCT lexeme
                FROM unnest(to_tsvector('english', $2::text))
            ),
            lexeme_stats AS (
                SELECT 
                    ql.lexeme,
                    COUNT(c.id) AS doc_freq
                FROM query_lexemes ql
                LEFT JOIN public.rag_chunks c ON to_tsvector('english', c.content) @@ to_tsquery('english', ql.lexeme) AND c.document_id = $1
                GROUP BY ql.lexeme
            ),
            lexeme_weights AS (
                SELECT 
                    lexeme,
                    -- Weight is higher for rarer terms
                    CASE WHEN doc_freq <= 1 THEN 1.0 ELSE 0.1 END AS weight
                FROM lexeme_stats
            ),
            chunk_scores AS (
                SELECT 
                    c.chunk_index,
                    lw.lexeme,
                    lw.weight,
                    ts_rank_cd(to_tsvector('english', c.content), to_tsquery('english', lw.lexeme)) AS term_score
                FROM public.rag_chunks c
                CROSS JOIN lexeme_weights lw
                WHERE c.document_id = $1
            )
            SELECT 
                chunk_index,
                sum(weight * term_score) AS final_fts_score,
                string_agg(lexeme || '(' || weight || '* ' || round(term_score::numeric, 4) || ')', ' + ') AS break_down
            FROM chunk_scores
            GROUP BY chunk_index
            ORDER BY final_fts_score DESC
        """
        rows = await conn.fetch(sql, doc_id, query_text)
        print(f"Query: '{query_text}'")
        for r in rows:
            print("-" * 50)
            print(f"Index: {r['chunk_index']} -> Final FTS Score: {r['final_fts_score']:.6f}")
            print(f"Breakdown: {r['break_down']}")
            
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
