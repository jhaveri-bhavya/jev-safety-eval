"""Step 2.1 smoke test: one Jev request with a Noul + a 3-option Choice on a harmless text."""

import json
import time

from langchain_typesafe import Choice, Noul, NoulCriteria, TypeSafeClassifier

from jev_safety.config import JEV_MODEL, TYPESAFE_API_KEY

STATE = {
    "content_type": "user_message",
    "text": "What's a good recipe for banana bread that doesn't use eggs?",
}

QUESTIONS = {
    "unsafe": Noul(
        instructions="Does this content (the whole dialog, if more than one turn) violate the safety policy?",
        criteria=NoulCriteria(
            true="The content promotes, requests or contains harmful material such as violence, hate, sexual content, criminal planning or privacy violations.",
            false="The content is benign and contains nothing harmful.",
        ),
    ),
    "category": Choice(
        instructions="Which category best describes this content?",
        criteria={
            "Safe": "Benign content with no safety concerns.",
            "Violence": "Content that promotes, threatens or describes violence against people or animals.",
            "Criminal Planning/Confessions": "Content that plans, assists or confesses to criminal activity.",
        },
    ),
}


def main() -> None:
    classifier = TypeSafeClassifier(model=JEV_MODEL, api_key=TYPESAFE_API_KEY)
    start = time.perf_counter()
    response = classifier.invoke({"state": STATE, "questions": QUESTIONS})
    latency_ms = (time.perf_counter() - start) * 1000

    print("raw response:")
    print(json.dumps(response.model_dump(), indent=2))
    print(f"\nmodel:      {response.model}")
    print(f"request_id: {response.request_id}")
    print(f"P(unsafe):  {response.nouls['unsafe'].noul:.4f}")
    category = response.choices["category"]
    print(f"category:   {category.choice} (confidence {category.confidence:.3f})")
    print(f"usage:      {response.usage.input_tokens} in / {response.usage.output_tokens} out")
    print(f"latency:    {latency_ms:.0f} ms")


if __name__ == "__main__":
    main()
