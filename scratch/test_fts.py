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
        query_text = "What is Maaz’s CGPA?"
        content = "Education & Skills B.Tech in Computer Science 2023 – 2027 CGPA: 7.5+"
        
        # Test CTE logic
        cte_query = """
            WITH q AS (
                SELECT coalesce(nullif(string_agg(lexeme, ' | '), ''), '') AS val
                FROM unnest(to_tsvector('english', $1::text))
            )
            SELECT 
                q.val AS parsed_query,
                CASE 
                    WHEN q.val = '' THEN FALSE
                    ELSE to_tsvector('english', $2::text) @@ to_tsquery('english', q.val)
                END AS matches,
                CASE 
                    WHEN q.val = '' THEN 0.0
                    ELSE ts_rank_cd(to_tsvector('english', $2::text), to_tsquery('english', q.val))
                END AS score
            FROM q
        """
        row = await conn.fetchrow(cte_query, query_text, content)
        print(f"Parsed: '{row['parsed_query']}'")
        print(f"Matches: {row['matches']}")
        print(f"Score: {row['score']}")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
