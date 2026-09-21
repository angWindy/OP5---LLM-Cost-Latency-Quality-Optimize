#!/usr/bin/env python3
"""
Optimization experiment P4: (LEGACY - SKIPPED).

After P3 we switched from SentenceTransformer to TF-IDF, where there is no
"question embedding" step -- TF-IDF computes question vector inline as part
of the same matrix transform. So P4 (batch question with sentences) is no
longer applicable.

This file is kept for the worklog record.

Result: SKIP (not applicable to TF-IDF pipeline).
"""
if __name__ == "__main__":
    print("P4 not applicable after P3 (TF-IDF used instead of SentenceTransformer).")
