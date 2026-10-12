def test_fusion():
    # Chunk 1: Maaz Ahmad (high vector, matches 'maaz')
    vec1 = 0.527267
    kw1 = 0.1
    
    # Chunk 2: CGPA (low vector, matches 'cgpa')
    vec2 = 0.3834
    kw2 = 0.1
    
    # Max keyword score is 0.1
    max_kw = 0.1
    
    # Formula 1: Weighted Average (current)
    # w_vec = 0.5, w_kw = 0.5
    f1_1 = (0.5 * vec1 + 0.5 * (kw1/max_kw)) / 1.0
    f1_2 = (0.5 * vec2 + 0.5 * (kw2/max_kw)) / 1.0
    print(f"Formula 1 (Weighted Avg): Chunk 1 = {f1_1:.4f}, Chunk 2 = {f1_2:.4f}")
    
    # Formula 2: Linear Boost (fused = vec + w_kw * norm_kw)
    f2_1 = vec1 + 0.5 * (kw1/max_kw)
    f2_2 = vec2 + 0.5 * (kw2/max_kw)
    print(f"Formula 2 (Linear Boost): Chunk 1 = {f2_1:.4f}, Chunk 2 = {f2_2:.4f}")
    
    # Formula 3: Reciprocal Rank Fusion (RRF)
    # Let's say Rank of Chunk 1: Vector = 1, Keyword = 1
    # Rank of Chunk 2: Vector = 2, Keyword = 1
    # RRF score = 1/(60 + r_vec) + 1/(60 + r_kw)
    rrf1 = 1/(60 + 1) + 1/(60 + 1)
    rrf2 = 1/(60 + 2) + 1/(60 + 1)
    print(f"RRF: Chunk 1 = {rrf1:.6f}, Chunk 2 = {rrf2:.6f}")

if __name__ == "__main__":
    test_fusion()
