# Crypto puzzle helper

This repository documents the steps needed to solve the small crypto puzzle that
originally accompanied the Leon Battista Alberti image. All of the raw clues are
stored in `data.txt`:

* the list of GitHub repository URLs whose initials spell out a Vigenère
  ciphertext,
* the dashed pattern that indicates how to segment the proclamation once it has
  been decrypted, and
* the plain hint `decode Great ;)` plus the hidden image clue that exposes the
  word `wind`.

The solving path is short:

1. **Alpha stage – hidden image text.** Combine the RGB channels from the image
   referenced by the puzzle to uncover the word `wind`. That confirms you are on
   the right track but it is only flavour for the next step.
2. **Vigenère stage – proclamation decode.** Take the first letters of each
   repository slug listed in `data.txt` to build the ciphertext
   `ciqjbqdpapidccc`. Decrypting it with the provided key `great` yields the
   phrase `hideandseek`. Apply the same key to Alberti's proclamation (the
   quoted text in `data.txt`) to obtain the sentence `doureallythinkitsallnetli`
   `fyapp`.
3. **Final message → Netlify checkpoint.** Format the proclamation using the
   dashed pattern from `data.txt` or the helper script below to read the final
   question:

   ```
   do u really think its all netlify app?
   ```

   Removing the spaces (and the question mark) yields the domain name
   `doureallythinkitsall.netlify.app`, which hosts the continuation of the
   puzzle. The landing page confirms the earlier alpha/beta/gamma hints and
   reveals three numbered clues (labelled with the Greek letters) together with
   a visible prime `11`, a hidden `gamma` prime `47`, and an instruction to
   “multiply all three and add .netlify.app.” The numeric attributes associated
   with the images provide the remaining factors:

   * Alpha image → `data-alpha="177"` (prime factors 3 × 59).
   * Beta image → `data-beta="198"` (prime factors 2 × 3² × 11) plus the visible
     `11` text block.
   * Gamma metadata → `data-gamma="47"` (already prime).

   Selecting the appropriate prime from each stage, multiplying them, and
   appending `.netlify.app` yields the next Netlify subdomain. The page also
   repeats the “wind guild” pointer, nudging players toward the Telegram handle
   shown in the coordinates block, and hides extra flavour text via Base64
   (`The cycle begins anew`, `I think something is wrong`).

The `solve.py` helper automates the bookkeeping:

* parses `data.txt` to recover the URL list, dash pattern, key, hidden hint, and
  proclamation ciphertext,
* demonstrates the Vigenère decoding using the supplied key, and
* prints the proclamation segmented by the dash pattern so the final question is
  easy to read,
* derives the Netlify hostname hidden in that question, and
* optionally downloads the follow-up Netlify page and extracts the new alpha,
  beta, and gamma hints (including their prime factors).

Run it with:

```bash
python solve.py
```

The optional `wordfreq` dependency can still be installed to improve the scoring
functions used by the substitution-solver experiments, but it is not required to
reach the finale described above.
