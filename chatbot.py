"""
This is my chatbot. I built it so it answers questions using ONLY the notes a
teacher gives it. What I really wanted to avoid was a bot that makes things up,
so if it isn't confident the answer is actually in the notes, I have it back off
and say it doesn't know instead of guessing.

To run it I just do:   python chatbot.py
"""
from pathlib import Path
from transformers import pipeline

# This is the most important number in the whole project, so I pulled it out to
# the top. If the model's confidence drops below this, I make the bot refuse
# instead of answering. That single rule is what stops it from inventing facts.
# I like to change it (try 0.05 or 0.4) to watch how the behaviour shifts.
ABSTAIN_THRESHOLD = 0.15

# First I read every .txt file in the "documents" folder and glue them together
# into one big block of notes. This is the only knowledge the bot is allowed to
# use, so if I want it to know something new, I just drop another file in there.
notes = ""
for file in Path("documents").glob("*.txt"):
    notes += file.read_text(encoding="utf-8") + "\n"

# Then I load the question-answering model. I do this once, up here, because it
# downloads the first time and I don't want to reload it for every question.
print("Loading the model...")
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")


def ask(question):
    """I hand the model one question plus my notes, and it either pulls the
    answer straight out of the notes or, if it's not confident enough, tells me
    it couldn't find it."""
    result = qa_model(question=question, context=notes)
    if result["score"] < ABSTAIN_THRESHOLD:
        return "I couldn't find that in the notes."
    return result["answer"]


# Finally I keep asking for questions in a loop so I can chat with it, and I stop
# when I type "quit".
if __name__ == "__main__":
    print("Ask me about the notes (type 'quit' to stop).")
    while True:
        question = input("\n> ")
        if question.lower() == "quit":
            break
        print(ask(question))
