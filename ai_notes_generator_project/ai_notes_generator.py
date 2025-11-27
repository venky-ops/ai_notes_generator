"""
ai_notes_generator.py

Simple AI Notes Generator:
- Reads a plain text file or a string
- Cleans and chunks notes
- Summarizes each chunk using HuggingFace transformers summarization pipeline
- Creates a heading and bullet points per chunk
- Outputs Markdown-formatted notes

Usage:
    python ai_notes_generator.py --input notes.txt --output cleaned_notes.md
"""

import argparse
import re
import math
from typing import List, Tuple
from transformers import pipeline, Pipeline
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# Ensure NLTK punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

##########################
# Utilities / Preprocess #
##########################

def clean_text(text: str) -> str:
    """Basic cleaning: normalize whitespace and remove repeated noisy chars."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove weird repeated characters (e.g., ----- or =====)
    text = re.sub(r'[-=]{3,}', '\n', text)
    # Remove multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # Trim spaces
    text = '\n'.join(line.strip() for line in text.splitlines())
    # Remove multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def chunk_text_by_paragraphs(text: str, max_chars: int = 1200) -> List[str]:
    """
    Split text into chunks for summarizer.
    - Splits on blank lines (paragraphs), then merges until max_chars is reached.
    """
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = []
    current_len = 0
    for p in paras:
        if current_len + len(p) + 2 > max_chars and current:
            chunks.append('\n\n'.join(current))
            current = [p]
            current_len = len(p)
        else:
            current.append(p)
            current_len += len(p) + 2
    if current:
        chunks.append('\n\n'.join(current))
    return chunks

##########################
# Summarization / LLM API#
##########################

def get_summarizer(model_name: str = "sshleifer/distilbart-cnn-12-6") -> Pipeline:
    """
    Create a summarization pipeline. You may switch model_name to a larger model if desired.
    """
    summarizer = pipeline("summarization", model=model_name, truncation=True)
    return summarizer

def summarize_chunk(summarizer: Pipeline, text: str, max_length: int = 120, min_length: int = 30) -> str:
    """
    Summarize one chunk. Some summarizers accept lists; handle exceptions.
    """
    # The pipeline expects not-too-long inputs; chunking handled earlier.
    try:
        out = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        summary = out[0]['summary_text'].strip()
    except Exception as e:
        # Fall back to returning a shorter slice if summarizer fails
        summary = text.strip()
    return summary

##########################
# Headline / Bullets     #
##########################

def generate_heading(text: str, max_words: int = 6) -> str:
    """
    Create a short heading from the first sentence or top phrase.
    """
    sents = sent_tokenize(text)
    if not sents:
        return "Summary"
    first = sents[0].strip()
    # Keep first N words
    words = first.split()
    heading = " ".join(words[:max_words])
    # Clean punctuation at end
    heading = re.sub(r'[^\w\s\-:]+$', '', heading)
    return heading.capitalize()

def bullets_from_text(text: str, max_bullets: int = 5) -> List[str]:
    """
    Turn text into concise bullets by splitting into sentences and trimming.
    """
    sents = sent_tokenize(text)
    bullets = []
    for s in sents:
        s = s.strip()
        # Keep short sentences as bullets; otherwise shorten
        if len(s) > 140:
            s = s[:140].rsplit(' ', 1)[0] + '...'
        bullets.append(s)
        if len(bullets) >= max_bullets:
            break
    return bullets

##########################
# Main generator         #
##########################

def generate_notes(text: str, summarizer: Pipeline,
                   chunk_max_chars: int = 1200,
                   summary_max_len: int = 120,
                   summary_min_len: int = 30,
                   bullets_per_chunk: int = 4) -> str:
    """
    Convert raw notes to formatted markdown notes.
    """
    text = clean_text(text)
    chunks = chunk_text_by_paragraphs(text, max_chars=chunk_max_chars)
    md_parts = []
    for idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        summary = summarize_chunk(summarizer, chunk, max_length=summary_max_len, min_length=summary_min_len)
        heading = generate_heading(summary)
        bullets = bullets_from_text(summary, max_bullets=bullets_per_chunk)
        md = []
        md.append(f"### {heading}\n")
        for b in bullets:
            md.append(f"- {b}")
        md.append("")  # blank line
        md.append(f"**Summary:** {summary}\n")
        md_parts.append("\n".join(md))
    final_md = "\n---\n\n".join(md_parts)
    return final_md

##########################
# CLI                    #
##########################

def main():
    parser = argparse.ArgumentParser(description="AI Notes Generator")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input text file containing rough notes.")
    parser.add_argument("--output", "-o", type=str, required=False, help="Path to output markdown file. If omitted, prints to stdout.")
    parser.add_argument("--model", "-m", type=str, default="sshleifer/distilbart-cnn-12-6", help="HF model for summarization.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw = f.read()

    summarizer = get_summarizer(args.model)
    result_md = generate_notes(raw, summarizer)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result_md)
        print(f"Saved formatted notes to {args.output}")
    else:
        print(result_md)


if __name__ == "__main__":
    main()
