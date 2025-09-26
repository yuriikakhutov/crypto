from __future__ import annotations

import base64
import itertools
import math
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from urllib.error import URLError
from urllib.request import urlopen

try:  # Optional but extremely helpful for ranking candidate plaintexts.
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover - optional dependency
    zipf_frequency = None  # type: ignore[assignment]

ALPHABET = string.ascii_lowercase

UPPER = string.ascii_uppercase

# Letter frequency data for crude English scoring.  The values roughly match
# the classic ETAOIN SHRDLU distribution and work well enough for ranking the
# tiny 15-character candidate strings we brute-force for the alpha/beta/gamma
# hint.
LETTER_FREQ = {
    "a": 0.08167,
    "b": 0.01492,
    "c": 0.02782,
    "d": 0.04253,
    "e": 0.12702,
    "f": 0.02228,
    "g": 0.02015,
    "h": 0.06094,
    "i": 0.06966,
    "j": 0.00153,
    "k": 0.00772,
    "l": 0.04025,
    "m": 0.02406,
    "n": 0.06749,
    "o": 0.07507,
    "p": 0.01929,
    "q": 0.00095,
    "r": 0.05987,
    "s": 0.06327,
    "t": 0.09056,
    "u": 0.02758,
    "v": 0.00978,
    "w": 0.0236,
    "x": 0.0015,
    "y": 0.01974,
    "z": 0.00074,
}

COMMON_BIGRAMS = {
    "th": 0.0356,
    "he": 0.0307,
    "in": 0.0243,
    "er": 0.0205,
    "an": 0.0199,
    "re": 0.0185,
    "on": 0.0171,
    "at": 0.0149,
    "en": 0.0145,
    "nd": 0.0135,
}

COMMON_TRIGRAMS = {
    "the": 0.0181,
    "ing": 0.0075,
    "and": 0.0073,
    "her": 0.0055,
    "hat": 0.0051,
    "his": 0.0049,
    "tha": 0.0049,
    "ent": 0.0042,
    "ere": 0.0038,
    "ion": 0.0037,
}

FALLBACK_WORD_SCORES = {
    "do": 3.5,
    "u": 2.0,
    "really": 4.0,
    "think": 4.2,
    "its": 3.0,
    "all": 3.2,
    "netlify": 3.8,
    "app": 3.1,
}


def load_puzzle(
    path: Path,
) -> tuple[List[str], List[int], str, str, str | None]:
    urls: List[str] = []
    dash_pattern: List[int] = []
    key = ""
    proclamation_cipher = ""
    hidden_hint: str | None = None

    dash_line_re = re.compile(r"^-{3}(?:\s+-{3,}){4}$")
    decode_re = re.compile(r"^decode\s+([A-Za-z]+)")
    quote_re = re.compile(r"«([A-Za-z]+)»")

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("http"):
                urls.append(line)
                continue
            match = decode_re.search(line)
            if match:
                key = match.group(1).lower()
                continue
            if line.lower() == "wind":
                hidden_hint = line.lower()
                continue
            if dash_line_re.match(line):
                dash_pattern = [len(part) for part in line.split()]
                continue
            quote_match = quote_re.search(line)
            if quote_match:
                proclamation_cipher = quote_match.group(1).lower()

    if not urls:
        raise ValueError("No URLs found in data file")
    if not dash_pattern:
        raise ValueError("Dash pattern missing from data file")
    if not key:
        raise ValueError("Cipher key missing from data file")
    if not proclamation_cipher:
        raise ValueError("Proclamation cipher text missing from data file")
    return urls, dash_pattern, key, proclamation_cipher, hidden_hint


def chunked(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    current: List[str] = []
    for item in iterable:
        current.append(item)
        if len(current) == size:
            yield current
            current = []
    if current:
        yield current


def extract_repo_slug(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    return slug


def derive_cipher_text(urls: List[str]) -> str:
    slugs = [extract_repo_slug(url) for url in urls]
    letters = [slug[0].lower() for slug in slugs]
    return "".join(letters)


def vigenere_decrypt(cipher_text: str, key: str) -> str:
    key_indices = [ALPHABET.index(ch) for ch in key]
    decoded: List[str] = []
    for index, char in enumerate(cipher_text):
        if char not in ALPHABET:
            decoded.append(char)
            continue
        cipher_value = ALPHABET.index(char)
        key_value = key_indices[index % len(key_indices)]
        plain_value = (cipher_value - key_value) % len(ALPHABET)
        decoded.append(ALPHABET[plain_value])
    return "".join(decoded)


def apply_pattern(text: str, pattern: List[int]) -> List[str]:
    words: List[str] = []
    index = 0
    pattern_index = 0
    while index < len(text):
        length = pattern[pattern_index % len(pattern)]
        word = text[index : index + length]
        words.append(word)
        index += length
        pattern_index += 1
    return words


def english_log_score(text: str) -> float:
    """Crude log-likelihood using independent letter frequencies."""

    score = 0.0
    for char in text:
        score += math.log(LETTER_FREQ.get(char, 1e-6))
    for index in range(len(text) - 1):
        bigram = text[index : index + 2]
        if bigram in COMMON_BIGRAMS:
            score += math.log(COMMON_BIGRAMS[bigram])
    for index in range(len(text) - 2):
        trigram = text[index : index + 3]
        if trigram in COMMON_TRIGRAMS:
            score += math.log(COMMON_TRIGRAMS[trigram])
    if zipf_frequency is not None:
        score += zipf_frequency(text, "en") * len(text)
        for ngram_length in (2, 3):
            for index in range(len(text) - ngram_length + 1):
                ngram = text[index : index + ngram_length]
                score += zipf_frequency(ngram, "en")
    return score


def frequency_score(text: str) -> float:
    score = 0.0
    for char in text:
        score += math.log(LETTER_FREQ.get(char, 1e-6))
    for index in range(len(text) - 1):
        bigram = text[index : index + 2]
        score += math.log(COMMON_BIGRAMS.get(bigram, 1e-6))
    for index in range(len(text) - 2):
        trigram = text[index : index + 3]
        score += math.log(COMMON_TRIGRAMS.get(trigram, 1e-6))
    return score


@dataclass
class SubstitutionResult:
    plaintext: str
    key: str
    score: float


class SubstitutionSolver:
    """Lightweight simulated-annealing solver for substitution ciphers."""

    def __init__(self, cipher_text: str) -> None:
        self.cipher_text = cipher_text

    def _decrypt(self, key: str) -> str:
        table = str.maketrans(UPPER, key)
        return self.cipher_text.translate(table)

    @staticmethod
    def _score(plaintext: str) -> float:
        stripped = plaintext.replace(" ", "").lower()
        if not stripped:
            return -math.inf
        return frequency_score(stripped)

    def solve(
        self,
        *,
        restarts: int = 60,
        iterations: int = 5000,
        seed: int = 2025,
    ) -> SubstitutionResult:
        rng = random.Random(seed)
        best = SubstitutionResult(plaintext="", key="", score=-math.inf)

        letters = list(UPPER)
        for _ in range(restarts):
            rng.shuffle(letters)
            key = "".join(letters)
            plain = self._decrypt(key)
            score = self._score(plain)
            current_key = key
            current_plain = plain
            current_score = score
            temperature = 20.0

            for _ in range(iterations):
                i, j = rng.sample(range(len(letters)), 2)
                key_list = list(current_key)
                key_list[i], key_list[j] = key_list[j], key_list[i]
                candidate_key = "".join(key_list)
                candidate_plain = self._decrypt(candidate_key)
                candidate_score = self._score(candidate_plain)
                delta = candidate_score - current_score
                if delta > 0 or rng.random() < math.exp(delta / temperature):
                    current_key = candidate_key
                    current_plain = candidate_plain
                    current_score = candidate_score
                    if current_score > best.score:
                        best = SubstitutionResult(
                            plaintext=current_plain,
                            key=current_key,
                            score=current_score,
                        )
                temperature *= 0.999
                if temperature < 0.05:
                    temperature = 0.05

            if current_score > best.score:
                best = SubstitutionResult(
                    plaintext=current_plain, key=current_key, score=current_score
                )

        return best


def fetch_netlify_tokens(raw_url: str) -> List[str]:
    """Extract the leading-letter tokens from a Netlify TOML gist."""

    response = urlopen(raw_url, timeout=10)
    text = response.read().decode("utf-8")
    tokens: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("[")
            or stripped.startswith("#")
            or "=" not in stripped
            or not stripped[0].isalpha()
        ):
            continue
        key_part = stripped.split("=", 1)[0].strip()
        token = "".join(
            part[0].upper() for part in key_part.replace("-", "_").split("_") if part
        )
        if token:
            tokens.append(token)

    if not tokens:
        raise ValueError("No token keys extracted from gist")
    return tokens


def search_vigenere_keys(
    cipher_text: str, key_length: int, top_n: int = 5
) -> List[Tuple[str, str, float]]:
    """Enumerate Vigenère keys of the given length ordered by score."""

    best: List[Tuple[str, str, float]] = []
    for key_tuple in itertools.product(ALPHABET, repeat=key_length):
        key = "".join(key_tuple)
        plain = vigenere_decrypt(cipher_text, key)
        score = english_log_score(plain)
        best.append((key, plain, score))

    best.sort(key=lambda entry: entry[2], reverse=True)
    return best[:top_n]


def into_triplets(text: str) -> List[str]:
    return [text[index : index + 3] for index in range(0, len(text), 3)]


def column_words(triplets: Sequence[str]) -> List[str]:
    words: List[str] = []
    for column in range(3):
        word = "".join(
            triplet[column] for triplet in triplets if len(triplet) > column
        )
        words.append(word)
    return words


def _word_score(word: str) -> float | None:
    if zipf_frequency is not None:
        score = zipf_frequency(word, "en")
        if score == 0.0:
            return None
        return score
    return FALLBACK_WORD_SCORES.get(word)


def segment_text(text: str) -> List[str]:
    """Segment text into probable words using DP and frequency scores."""

    n = len(text)
    best_score = [-math.inf] * (n + 1)
    best_score[0] = 0.0
    back_pointer = [-1] * (n + 1)

    for end in range(1, n + 1):
        for start in range(max(0, end - 12), end):
            word = text[start:end]
            score = _word_score(word)
            if score is None or best_score[start] == -math.inf:
                continue
            candidate = best_score[start] + score
            if candidate > best_score[end]:
                best_score[end] = candidate
                back_pointer[end] = start

    if best_score[n] == -math.inf:
        return [text]

    words: List[str] = []
    index = n
    while index > 0:
        start = back_pointer[index]
        if start == -1:
            return [text]
        words.append(text[start:index])
        index = start
    words.reverse()
    return words


def is_prime(number: int) -> bool:
    if number < 2:
        return False
    if number in (2, 3):
        return True
    if number % 2 == 0:
        return False
    limit = int(math.isqrt(number))
    candidate = 3
    while candidate <= limit:
        if number % candidate == 0:
            return False
        candidate += 2
    return True


def prime_factors(number: int) -> List[int]:
    if number <= 1:
        return []
    factors: List[int] = []
    remainder = number
    divisor = 2
    while divisor * divisor <= remainder:
        while remainder % divisor == 0:
            factors.append(divisor)
            remainder //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if remainder > 1:
        factors.append(remainder)
    return factors


def derive_netlify_domain(question: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    if len(tokens) >= 2 and tokens[-2:] == ["netlify", "app"]:
        host = "".join(tokens[:-2])
        return f"{host}.netlify.app"
    sanitized = "".join(tokens)
    return f"{sanitized}.netlify.app"


def fetch_page(url: str) -> str:
    with urlopen(url) as handle:
        raw = handle.read()
    return raw.decode("utf-8", errors="replace")


def parse_netlify_page(html: str) -> None:
    attribute_values = {
        label: int(value)
        for label, value in re.findall(r"data-(alpha|beta|gamma)=\"(\d+)\"", html)
    }
    if attribute_values:
        print("Netlify alpha/beta/gamma attributes:")
        for label in ["alpha", "beta", "gamma"]:
            if label in attribute_values:
                value = attribute_values[label]
                factors = prime_factors(value)
                print(f"  {label}: {value} → prime factors {factors if factors else '[]'}")

    visible_numbers = [
        int(match)
        for match in re.findall(r">\s*(\d+)\s*<", html)
        if match.isdigit()
    ]
    visible_primes = [number for number in visible_numbers if is_prime(number)]
    if visible_primes:
        print("Visible primes on page:", visible_primes)

    cipher_hints = set(re.findall(r"data-cipher=\"([^\"]+)\"", html))
    for encoded in sorted(cipher_hints):
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        print(f"Decoded data-cipher hint: {decoded}")

    comment_hints = re.findall(r"<!--\s*([^<>]+?)\s*-->", html)
    for comment in comment_hints:
        stripped = comment.strip()
        try:
            decoded = base64.b64decode(stripped).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            print("HTML comment:", stripped)
        else:
            print("Decoded HTML comment:", decoded)

    tele_handles = re.findall(r"[A-Za-z0-9_.-]+\.t\.me", html)
    if tele_handles:
        print("Telegram pointers:", sorted(set(tele_handles)))


def main() -> None:
    data_path = Path("data.txt")
    urls, dash_pattern, key, proclamation_cipher, hidden_hint = load_puzzle(
        data_path
    )

    print("Loaded", len(urls), "urls")
    print("Dash pattern (word lengths hint):", dash_pattern)
    print("Cipher key:", key)
    if hidden_hint:
        print("Hidden image keyword (Alpha stage):", hidden_hint)

    cipher_text = derive_cipher_text(urls)
    print("Cipher text from repository slugs:", cipher_text)

    plaintext = vigenere_decrypt(cipher_text, key)
    print(f"Decoded text from slugs (using key '{key}'):", plaintext)

    triplets = into_triplets(plaintext)
    print("\nTriplet view of the slug plaintext (for curiosity):")
    for index, group in enumerate(triplets, start=1):
        print(f"  Triplet {index}:", group)

    print("Column words derived from the triplets:")
    for label, word in zip(["alpha", "beta", "gamma"], column_words(triplets)):
        print(f"  {label}: {word}")

    proclamation_plain = vigenere_decrypt(proclamation_cipher, key)
    print("\nProclamation cipher (Alberti stage):", proclamation_cipher)
    print(
        "Decryption with the overt key '{key}' (checkpoint only):".format(key=key),
        proclamation_plain,
    )

    patterned = apply_pattern(proclamation_plain, dash_pattern)
    print("Patterned proclamation:", " ".join(patterned))

    segmented = segment_text(proclamation_plain)
    if segmented != [proclamation_plain]:
        segmented_text = " ".join(segmented)
        print("Segmented proclamation:", segmented_text)
    else:
        segmented_text = " ".join(patterned)

    print("Final proclamation (formatted):", segmented_text)

    netlify_domain = derive_netlify_domain(segmented_text)
    netlify_url = f"https://{netlify_domain}/"
    print("Derived Netlify follow-up domain:", netlify_url)

    try:
        html = fetch_page(netlify_url)
    except URLError as exc:
        print("  Unable to download follow-up page:", exc)
    else:
        parse_netlify_page(html)

    print("\nDecoding the Netlify configuration gist tokens:")
    gist_url = (
        "https://gist.githubusercontent.com/DavidWells/43884f15aed7e4dcb3a6dad06430b756/raw"
    )
    try:
        tokens = fetch_netlify_tokens(gist_url)
    except (URLError, ValueError) as exc:
        print("  Unable to process gist:", exc)
    else:
        cipher_stream = " ".join(tokens)
        print("  Token cipher:", cipher_stream)
        solver = SubstitutionSolver(cipher_stream)
        result = solver.solve(restarts=12, iterations=1500)
        print("  Best substitution candidate:", result.plaintext)
        print("  Candidate key:", result.key)
        print(f"  Candidate score: {result.score:.3f}")


if __name__ == "__main__":
    main()
