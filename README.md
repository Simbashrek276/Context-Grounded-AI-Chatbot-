# Context-Grounded-AI-Chatbot

This is a small chatbot I built that answers questions using **only** the notes
you give it, and honestly says *"I couldn't find that"* when the answer isn't in
those notes. I made it this way on purpose: most chatbots will happily make up a
confident-sounding wrong answer, and if you're a student trusting it to study,
that's a real problem. Mine either quotes the notes or admits it doesn't know.

Under the hood it's DistilBERT (extractive question answering) plus one
confidence threshold that decides when to answer and when to back off.

## Being honest about what this is (and isn't)

Before anything else — this is a learning project, and it has real limits I kept
bumping into. I'd rather put them up front than pretend it's smarter than it is:

- **It can't actually reason.** It's *extractive*, which means it copies a span
  of text straight out of the notes. It can't paraphrase, do arithmetic, combine
  two facts, or explain anything in its own words. If the answer isn't already
  sitting in the notes as a phrase, it simply can't produce it.
- **Confident doesn't mean correct.** The whole design leans on the confidence
  score, but the model is sometimes confidently wrong. On my 20 questions it gets
  18 — and the two it misses, it answers *wrongly but with high confidence*, so
  the threshold never catches them. That's the honest weak spot of the approach.
- **The threshold is hand-tuned.** I landed on 0.15 by trying values against my
  own questions. It works for my notes; it's not a magic number, and a different
  set of notes would probably need a different one.
- **The test set is tiny.** Twenty questions is enough to see the idea working,
  not enough to claim a real accuracy figure. Treat "18/20" as a demo, not a
  benchmark.
- **It's picky about wording.** Ask the same thing a different way and the
  confidence can swing. The model is small and a few years old, so it matches
  patterns more than it understands meaning.

None of that breaks the core idea — refusing when it's unsure — but it's the gap
between a class project like this and something you'd actually rely on.

## The idea, plainly

1. You put a teacher's notes (plain `.txt` files) into the `documents/` folder.
   You can add as many topics as you like — I use three (Vietnamese history, the
   Solar System, and photosynthesis).
2. When you ask a question, the model looks through those notes and pulls out the
   piece of text that answers it, along with a confidence score.
3. If that confidence is too low, the bot refuses instead of guessing. That
   "refuse when unsure" rule is the whole point.

## What's in here

| File | What it's for |
|------|---------------|
| `chatbot.py` | The chatbot itself. You run this to ask it questions. |
| `confidence_scores.py` | The evaluation. You run this to test the bot on 20 questions, see every confidence score, get an overall accuracy, and save a chart. |
| `confidence_chart.png` | The chart the evaluation saves: each question's confidence, coloured by whether it's answerable. |
| `documents/` | The notes the bot is allowed to use. Drop your own `.txt` files here. |

## How to run it

You need Python and the libraries in `requirements.txt`. Install them once:

```bash
pip install -r requirements.txt
```

Then, to chat with it, you run:

```bash
python chatbot.py
```

It loads the model and waits. You type a question, and it either answers from the
notes or tells you it couldn't find it. Type `quit` to stop.

## Testing how good it is

This is the part I care most about. To evaluate the bot, you run:

```bash
python confidence_scores.py
```

That asks 20 questions — sixteen answerable from the documents (across three
topics) and four that aren't anywhere in them — and for each one prints the
model's **confidence score**, the answer it picked, and whether that was correct.
It finishes with an overall accuracy, a threshold sweep, and it saves
`confidence_chart.png`.

On my documents it scores **18/20**: it answers the covered questions and refuses
all four it shouldn't know. As I mentioned above, the two misses are questions it
answers *confidently but wrongly* — that's exactly where this simple approach
runs out of road. The chart still makes the main result clear: every answerable
question sits well above the threshold, while all four out-of-scope ones sit far
below it.

## Where the confidence score comes from

When the model answers, it returns a dictionary like
`{'score': 0.917, 'answer': 'carbon dioxide', ...}`. The `score` is the
confidence — how sure the model is about where the answer begins and ends. That's
the exact value `chatbot.py` compares against its threshold, and the value
`confidence_scores.py` prints for every question.

## Making it answer about something else

You don't need to touch any code. You just add or replace the `.txt` files in
`documents/` with your own notes, update the questions in `confidence_scores.py`
to match, and run it again — the bot now answers from your material instead.
