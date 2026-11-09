import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Locate project files relative to this Python file.
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "docs" / "carbonx_knowledge_base.md"

# Load local environment variables from the project root.
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()


class ChatbotConfigurationError(Exception):
    """Raised when the chatbot is not configured correctly."""


class ChatbotProviderError(Exception):
    """Raised when the AI provider cannot generate an answer."""


def load_knowledge_sections():
    """Read the Markdown knowledge base and split it by headings."""
    if not KNOWLEDGE_BASE_PATH.is_file():
        raise ChatbotConfigurationError(
            "The CarbonX knowledge-base file is missing."
        )

    content = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")

    sections = []
    current_title = "Introduction"
    current_lines = []

    for line in content.splitlines():
        if line.startswith("#"):
            if current_lines:
                body = "\n".join(current_lines).strip()

                if body:
                    sections.append({
                        "title": current_title,
                        "content": body,
                    })

            current_title = line.lstrip("#").strip()
            current_lines = []

        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()

        if body:
            sections.append({
                "title": current_title,
                "content": body,
            })

    return sections


def tokenize(text):
    """Convert text into lowercase keywords."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve_sections(question, sections, limit=1):
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can",
        "do", "does", "for", "from", "how", "i", "in", "is", "it",
        "me", "of", "on", "or", "our", "the", "this", "to", "us",
        "was", "we", "what", "when", "where", "which", "who", "why",
        "with", "you", "your", "carbonx",
    }

    question_words = tokenize(question) - stop_words

    # Handle general project-identity questions separately.
    question_lower = question.lower().strip()
    if question_lower in {
        "what is carbonx",
        "what is carbonx?",
        "tell me about carbonx",
        "what is the carbonx project",
   }:
        for section in sections:
            if "project overview" in section["title"].lower():
                return [section]
    if not question_words:
        return []

    scored_sections = []

    for section in sections:
        title_words = tokenize(section["title"]) - stop_words
        content_words = tokenize(section["content"]) - stop_words

        title_matches = question_words.intersection(title_words)
        content_matches = question_words.intersection(content_words)

        # Title matches are more important than ordinary content matches.
        score = (8 * len(title_matches)) + (2 * len(content_matches))

        # Give extra weight to exact meaningful phrases/keywords.
        question_lower = question.lower()
        title_lower = section["title"].lower()
        content_lower = section["content"].lower()

        if "carbon credits" in question_lower:
            if "credit" in title_lower or "credit" in content_lower:
                score += 5

        if "simulated" in question_lower or "simulation" in question_lower:
            if "simulation" in title_lower or "simulated" in content_lower:
                score += 10

        if "verified" in question_lower or "verification" in question_lower:
            if "verification" in title_lower or "verified" in content_lower:
                score += 5

        if "benchmark" in question_lower or "benchmarking" in question_lower:
            if "benchmark" in title_lower or "benchmark" in content_lower:
                score += 5

        if "emission" in question_lower or "emissions" in question_lower:
            if "emission" in title_lower or "emissions" in content_lower:
                score += 5

        if "limitation" in question_lower or "limitations" in question_lower:
            if "limitation" in title_lower or "limitations" in content_lower:
                score += 5

        if question_lower.strip() in {
            "what is carbonx",
            "what is carbonx?",
            "tell me about carbonx",
            "what is the carbonx project",
        }:
            if "project overview" in title_lower:
                score += 20

        if score > 0:
            scored_sections.append((score, section))

    scored_sections.sort(key=lambda item: item[0], reverse=True)

    return [section for _, section in scored_sections[:limit]]


def answer_question(question):
    """Answer a CarbonX question using the retrieved knowledge base."""

    # ---------------------------------------------------------
    # 1. Check Gemini API configuration
    # ---------------------------------------------------------
    if not GEMINI_API_KEY:
        raise ChatbotConfigurationError(
            "The AI assistant is not configured. Set GEMINI_API_KEY "
            "in the local .env file."
        )

    # ---------------------------------------------------------
    # 2. Load CarbonX knowledge base
    # ---------------------------------------------------------
    try:
        sections = load_knowledge_sections()

    except ChatbotConfigurationError:
        raise

    except OSError:
        raise ChatbotConfigurationError(
            "The CarbonX knowledge base could not be read."
        ) from None

    # ---------------------------------------------------------
    # 3. Retrieve only the most relevant section
    # ---------------------------------------------------------
    selected_sections = retrieve_sections(
        question,
        sections,
        limit=1
    )

    # ---------------------------------------------------------
    # 4. Handle questions with no relevant information
    # ---------------------------------------------------------
    if not selected_sections:
        return (
            "I could not find relevant information in the CarbonX "
            "knowledge base to answer that question. Please ask about "
            "CarbonX features, emissions, benchmarking, credits, or "
            "documented project limitations."
        )

    # ---------------------------------------------------------
    # 5. Prepare context
    # ---------------------------------------------------------
    section = selected_sections[0]

    context = (
        f"Source section: {section['title']}\n"
        f"{section['content']}"
    )

    # ---------------------------------------------------------
    # 6. Prepare prompt
    # ---------------------------------------------------------
    prompt = f"""
You are the CarbonX project assistant.

IMPORTANT:
CarbonX refers ONLY to the CarbonX project described in the
supplied knowledge-base context.

Answer the user's question using ONLY the supplied CarbonX
knowledge-base section.

Rules:
- Do not use outside knowledge.
- Do not invent facts, numbers, database results, or capabilities.
- If the supplied context does not contain enough information,
  say so clearly.
- Keep the answer concise and easy to understand.
- Do not generate SQL.
- Do not modify database records.
- Do not claim to perform transactions.
- Distinguish simulated carbon credits from verified real-world credits.
- Do not claim independent verification unless the context explicitly
  states it.
- At the end, mention the source section used.

KNOWLEDGE-BASE CONTEXT:
{context}

USER QUESTION:
{question}
"""

    # ---------------------------------------------------------
    # 7. Create Gemini client
    # ---------------------------------------------------------
    # 15 seconds maximum for one Gemini request.
    # This prevents requests from hanging for several minutes.
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            timeout=15000
        ),
    )

    # ---------------------------------------------------------
    # 8. Retry temporary Gemini errors
    # ---------------------------------------------------------
    max_attempts = 3

    for attempt in range(max_attempts):

        try:
            print(
                f"Calling Gemini "
                f"(attempt {attempt + 1}/{max_attempts})..."
            )
            

            # -------------------------------------------------
            # 9. Generate answer
            # -------------------------------------------------
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1000,
                ),
            )

            # -------------------------------------------------
            # 10. Extract answer
            # -------------------------------------------------
            answer = response.text

            # -------------------------------------------------
            # 11. Check for empty answer
            # -------------------------------------------------
            if not answer or not answer.strip():
                raise ChatbotProviderError(
                    "The AI provider returned an empty response."
                )

            print("Gemini response received successfully.")

            # -------------------------------------------------
            # 12. Return answer
            # -------------------------------------------------
            return answer.strip()

        except ChatbotProviderError:
            raise

        except Exception as e:

            error_text = str(e)

            print(
                f"Gemini attempt {attempt + 1} failed:"
            )

            print(
                f"Error type: {type(e).__name__}"
            )

            print(
                f"Error: {error_text}"
            )

            # -------------------------------------------------
            # 13. Identify temporary errors
            # -------------------------------------------------
            is_transient = (
                "503" in error_text
                or "504" in error_text
                or "UNAVAILABLE" in error_text
                or "DEADLINE_EXCEEDED" in error_text
            )

            # -------------------------------------------------
            # 14. Stop for permanent errors
            # -------------------------------------------------
            if (
                not is_transient
                or attempt == max_attempts - 1
            ):
                raise ChatbotProviderError(
                    "The AI service is temporarily unavailable. "
                    "Please try again in a moment."
                ) from None

            # -------------------------------------------------
            # 15. Wait before retrying
            # -------------------------------------------------
            wait_time = 2 * (attempt + 1)

            print(
                f"Temporary Gemini error. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    # ---------------------------------------------------------
    # 16. Final fallback
    # ---------------------------------------------------------
    raise ChatbotProviderError(
        "The AI service is temporarily unavailable. "
        "Please try again in a moment."
    )