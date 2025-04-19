from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.language import Language
from spacy.util import filter_spans
from unidecode import unidecode
import os
import json
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Initialize SpaCy
nlp = spacy.blank("vi")
matcher = Matcher(nlp.vocab)

# Load environment variables and configure extraction method
load_dotenv()
EXTRACT_METHOD = os.getenv("EXTRACT_METHOD", "ner").lower()
# Initialize LLM for component type extraction
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
llm = ChatOllama(model=LLM_MODEL)
SYSTEM_TYPE_PROMPT = """
Extract from the given text:
- PC component type: a PC component has exact, same name as one of the following: CPU; Motherboard; RAM; GraphicsCard (as GPU); InternalHardDrive (as SSD or HDD); CPUCooler (as cooler); Case; PowerSupply
- Budget: numbers with units (triệu, vnd, đồng)
- Purpose: e.g., chơi game; thiết kế; văn phòng.
* Notice If GPU chipset detected (e.g., "NVIDIA RTX 3060 Ti"), return GPU
Respond ONLY with a JSON array of [value, label] pairs, where label is one of the component labels, or BUDGET, or PURPOSE.
Example: [["Intel i7-9700K","CPU"], ["15 triệu","BUDGET"], ["chơi game","PURPOSE"]]
No additional text.
"""

def preprocess_text(text):
    """Normalize text: lowercase, remove diacritics, and clean."""
    text = text.lower()
    text = unidecode(text)  # Normalize accented characters
    text = text.replace("  ", " ").strip()  # Remove extra spaces
    return text


def add_rules_to_matcher(matcher):
    """Add entity matching rules."""
    # Budget: Match numbers followed by currency or unit keywords
    matcher.add(
        "BUDGET",
        [
            [{"TEXT": {"REGEX": r"\d+([.,]\d+)?"}}, {"LOWER": {"IN": ["triệu", "trieu", "tr", "vnđ", "vnd", "đồng"]}}],
            [{"TEXT": {"REGEX": r"\d+([.,]\d+)?(triệu|trieu|tr|vnđ|vnd|đồng)"}}],
        ],
    )

    # Purpose
    matcher.add("PURPOSE", [[{"LOWER": {"IN": ["chơi game", "game", "gaming", "play game", "thiết kế", "văn phòng"]}}]])

    # CPU
    matcher.add(
        "CPU",
        [
            # Intel i3/i5/i7/i9 variations
            [{"TEXT": {"REGEX": r"(intel)"}}, {"TEXT": {"REGEX": r"i[3579]"}}, {"TEXT": {"REGEX": r"(-?\d{3,4}[a-z]{0,2})"}}],
            [{"TEXT": {"REGEX": r"i[3579]"}}, {"TEXT": {"REGEX": r"(-?\d{3,4}[a-z]{0,2})"}}],

            # AMD Ryzen variations
            [{"TEXT": {"REGEX": r"(amd|ryzen)"}}, {"TEXT": {"REGEX": r"([3579])"}}, {"TEXT": {"REGEX": r"(\d{3,4}[a-z]{0,3})"}}],
            [{"TEXT": {"REGEX": r"(ryzen\s?[3579])"}}, {"TEXT": {"REGEX": r"(\d{3,4}[a-z]{0,3})"}}],
            [{"TEXT": {"REGEX": r"(r\s?[3579])"}}, {"TEXT": {"REGEX": r"(\d{3,4}[a-z]{0,3})"}}],
        ]
    )

    # GPU
    matcher.add(
        "GPU",
        [
            # NVIDIA RTX and GTX variations
            [{"TEXT": {"REGEX": r"(nvidia|rtx|gtx|mx)"}}, {"TEXT": {"REGEX": r"(\d{3,4})"}}, {"TEXT": {"REGEX": r"(ti|super)?"}}],
            [{"TEXT": {"REGEX": r"(rtx|gtx|mx)\s?\d{3,4}(ti|super)?"}}],

            # AMD RX variations
            [{"TEXT": {"REGEX": r"(amd|rx)"}}, {"TEXT": {"REGEX": r"(\d{3,4})"}}, {"TEXT": {"REGEX": r"(xt|xtx)?"}}],
            [{"TEXT": {"REGEX": r"(rx)\s?\d{3,4}(xt|xtx)?"}}],
        ]
    )

    #RAM
    matcher.add(
        "RAM",
        [
            [
                {
                    "TEXT": {
                        "REGEX": r"(ram|memory)?\s*(\d+)\s*(gb|g|giga|gbx|gbx\d+|x\d+)?\s*(ddr[345])?\s*[- ]?(\d{3,4})?\s*(mhz)?(\s*ram|memory)?"
                    }
                }
            ]
        ],
    )

    return matcher


@Language.component("custom_entity_component")
def custom_entity_component(doc):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label=label) for match_id, start, end in matches for label in [nlp.vocab.strings[match_id]]]
    spans = filter_spans(spans)
    doc.ents = list(doc.ents) + spans
    return doc


# Add rules and pipeline only once during initialization
matcher = add_rules_to_matcher(matcher)
nlp.add_pipe("custom_entity_component", last=True)


def merge_related_entities(entities):
    """Post-process entities to merge related ones."""
    merged = []
    i = 0
    while i < len(entities):
        entity_text, label = entities[i]
        if label in {"CPU", "GPU", "RAM"}:
            while i + 1 < len(entities) and entities[i + 1][1] == label:
                entity_text += f" {entities[i + 1][0]}"
                i += 1
        merged.append((entity_text, label))
        i += 1

    return merged


def extract_entities(text):
    """Extract structured data from text."""
    text = preprocess_text(text)
    doc = nlp(text)

    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    # Merge related entities
    entities = merge_related_entities(entities)
    return entities


# FastAPI app
app = FastAPI()


class TextInput(BaseModel):
    text: str


@app.post("/extract")
async def extract_entities_api(input: TextInput):
    text = input.text
    if EXTRACT_METHOD == "llm":
        # Use LLM to extract component type
        messages = [
            {"role": "system", "content": SYSTEM_TYPE_PROMPT},
            {"role": "user", "content": text}
        ]
        llm_response = llm.invoke(messages)
        component_type = llm_response.content.strip() if hasattr(llm_response, "content") else str(llm_response).strip()
        # Extract JSON substring and parse
        raw = component_type
        start = raw.find('[')
        end = raw.rfind(']') + 1
        json_str = raw[start:end] if start >= 0 and end > start else raw
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback to Python literal
            import ast
            try:
                data = ast.literal_eval(raw)
            except Exception:
                data = [(text, component_type)]
        return {"data": data}
    else:
        entities = extract_entities(text)
        return {"data": entities}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)