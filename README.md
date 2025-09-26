# Crypto puzzle helper

This repository captures the reasoning process for a small cryptography-themed
puzzle.  The `data.txt` file stores the raw clues (GitHub links, the dashed
layout hint, the cipher key hint, and the proclamation pulled from an image).

`solve.py` reads those clues and reproduces the deduction steps:

1. Gather the repository slugs from the provided links.
2. Use the repeated dashed layout to extract the word-length pattern hint.
3. Detect the Vigenère keyword from the "decode" clue (currently `great`) and
   surface it in the CLI output so the message always reflects the parsed key.
4. Decrypt the string formed by the slug initials, display the alpha/beta/gamma
   triplets, and score all 3-letter keys hinted by "alpha, beta, gamma" to find
   stronger plaintexts.  The script prints the highest-ranked candidates rather
   than hard-coding a single example so the output always matches the search
   results for the current scoring heuristic.
5. Decrypt Leon Battista's proclamation, reshape the plaintext according to the
   dash pattern hint, automatically segment the message into natural words using
   frequency-based scoring, and surface the final question (with a trailing
   question mark) for quick reference.
6. Pull the Netlify configuration gist linked in the discussion, extract the
   leading letters of each configuration key, and run a simple substitution
   search to present the strongest candidate plaintext recovered from that
   token stream.

The optional `wordfreq` package, if available, improves the 3-letter key search
by providing richer English scoring.  Without it the solver falls back to a
lighter frequency heuristic.

The proclamation decodes to the plain-language question:

> **do u really think its all netlify app?**

Run the helper with:

```bash
python solve.py
```
