"""
This is how I evaluate my chatbot. For every test question I print the model's
confidence, the answer it picked, and whether that counts as correct -- then I
tally an overall score and sweep the abstention threshold to see how it behaves.

Where the confidence comes from: when I call the model with
    result = qa_model(question=q, context=notes)
it hands back a dictionary like {'score': 0.917, 'answer': 'carbon dioxide', ...}.
The 'score' is the confidence. Internally the model rates how likely each token is
to START the answer and how likely each is to END it; the pipeline turns those
into probabilities and multiplies the best start-probability by the best
end-probability. So the score is basically "how sure the model is about where the
answer begins and ends". It's the same number chatbot.py checks against its
threshold to decide whether to answer or refuse.

Run it:   python confidence_scores.py
"""
from pathlib import Path
from transformers import pipeline

# Same threshold chatbot.py uses. below this confidence, the bot refuses.
ABSTAIN_THRESHOLD = 0.15

# I read the same notes the chatbot uses, which is every .txt file in the folder.
notes = ""
for file in Path("documents").glob("*.txt"):
    notes += file.read_text(encoding="utf-8") + "\n"

print("Loading the model...")
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# My test set. Each item is (question, the answer I expect). I spread the
# questions across all three documents so I'm testing across topics. The ones
# with None as the expected answer are NOT in any document on purpose -- the bot
# should refuse them, so "correct" for those means it abstained.
tests = [

    ("Who ruled the kingdom of Van Lang?", "hung kings"),
    ("When did the Vietnam War end?", "1975"),
    ("Who led Vietnam to independence at the Battle of Bach Dang?", "ngo quyen"),
    ("When was the Temple of Literature established?", "1070"),
    ("Which dynasty repelled the Mongol invasions?", "tran"),
    ("When did French colonial rule in Vietnam end?", "1945"),

    ("What is the largest planet in the Solar System?", "jupiter"),
    ("Which planet is closest to the Sun?", "mercury"),
    ("Why is Mars called the Red Planet?", "iron oxide"),
    ("How old is the Solar System?", "4.6 billion"),
    ("How many planets are in the Solar System?", "eight"),
    ("What is Saturn known for?", "rings"),

    ("Which pigment absorbs light during photosynthesis?", "chlorophyll"),
    ("What gas do plants release during photosynthesis?", "oxygen"),
    ("Where does photosynthesis mainly take place?", "leaves"),
    ("What sugar do plants produce during photosynthesis?", "glucose"),

    ("What is the capital of France?", None),
    ("Who won the 2018 World Cup?", None),
    ("Who wrote the play Romeo and Juliet?", None),
    ("What is the boiling point of water?", None),
]

# I ask the model each question once and remember the result, so I don't have to
# run it again for the threshold sweep below.
results = []
for question, expected in tests:
    r = qa_model(question=question, context=notes)
    results.append((question, expected, r["score"], r["answer"]))


def is_correct(expected, score, answer, threshold):
    """Did the bot do the right thing at this threshold?"""
    refused = score < threshold
    if expected is None:
        return refused       # should have refused
    return (not refused) and expected in answer.lower()# should have answered correctly


# First I print a row per question at the normal threshold.
print(f"\n{'score':>6} | {'verdict':<8} | {'answer':<22} | question")
print("-" * 90)
correct = answered = 0
for question, expected, score, answer in results:
    refused = score < ABSTAIN_THRESHOLD
    if not refused:
        answered += 1
    ok = is_correct(expected, score, answer, ABSTAIN_THRESHOLD)
    if ok:
        correct += 1
    verdict = "CORRECT" if ok else "WRONG"
    shown = "(refused)" if refused else answer
    print(f"{score:>6.3f} | {verdict:<8} | {shown:<22} | {question}")

print("-" * 90)
print(f"Score at threshold {ABSTAIN_THRESHOLD}: {correct}/{len(tests)} correct "
      f"({answered} answered, {len(tests) - answered} refused)")

# Then I sweep the threshold to see the trade-off: too low and it answers things
# it shouldn't; too high and it refuses answers it actually got right.
print("\nThreshold sweep:")
for thr in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60]:
    hits = sum(is_correct(e, s, a, thr) for _, e, s, a in results)
    print(f"  threshold={thr:0.2f} -> {hits}/{len(tests)} correct")


# Finally I draw a simple chart and save it as a PNG: one dot per question, blue
# for answerable questions and orange for out-of-scope ones, with a dashed line
# at the abstain threshold. The gap between the two colours is the whole point.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

answerable = [(i + 1, s) for i, (_, e, s, _) in enumerate(results) if e is not None]
out_scope = [(i + 1, s) for i, (_, e, s, _) in enumerate(results) if e is None]

fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(ABSTAIN_THRESHOLD, color="gray", linestyle="--", label=rf"threshold $\tau$ = {ABSTAIN_THRESHOLD}")
ax.scatter(*zip(*answerable), color="#0072B2", s=90, label="Answerable (in the notes)")
ax.scatter(*zip(*out_scope), color="#D55E00", s=90, label="Out-of-scope (not in the notes)")

ax.set_xlabel("Question number")
ax.set_ylabel("Model confidence")
ax.set_title("Confidence per question")
ax.set_ylim(0, 1.25)
ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig("confidence_chart.png", dpi=150)
print("\nSaved chart to confidence_chart.png")
