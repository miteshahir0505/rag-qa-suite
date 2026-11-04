# rag-qa-suite

A small search-and-answer system built on company documents, together with a set of tests that check each stage separately: how the documents are cut, whether search finds the right piece, whether the system admits when the documents do not answer, and whether the written answer stays faithful to the pieces it was given.

It is written in TypeScript. Search runs on your own PC (no key needed). The answer-writing and answer-checking AI models use the free Groq API.

## Why this project

A support chatbot that answers from documents can fail in quiet ways. The answer looks fine, but the cause is somewhere earlier. Here is a real case from this repo.

A customer asks: *"What happens if I try to buy something that is sold out?"* The documents answer this in one section ("Out of stock items"). But the search put a different section first: "Failed payments". When the writer was given only that first piece, it replied that the documents do not cover sold-out items. That is false, and it is not the writer's fault. The writer was honest about what it had been given. The search had handed it the wrong piece.

With the top three pieces, the right one was second and the writer answered correctly. So the test showed two things at once: why the writer should get more than one piece, and that this failure cannot be fixed by changing the writer's rules.

This project tests each stage on its own, so a failure can be traced to the stage that caused it.

## How the system works

1. **Cut** each document into pieces (called chunks).
2. **Store** each piece as a list of numbers that captures its meaning (an embedding).
3. **Find** the pieces whose meaning is closest to the customer's question.
4. **Write** an answer using only those pieces (an AI model, `openai/gpt-oss-20b`).
5. **Check** the answer against the pieces it was given (a second, bigger AI model, `openai/gpt-oss-120b`, acting as judge).

The sample documents are in `docs/`: three product requirement documents (add to cart, checkout, login), an API reference (orders) and one set of shipping notes.

## What was measured

### Cutting the documents

| Way of cutting | Pieces | Pieces that stop mid-sentence |
|---|---|---|
| By section heading | 22 | 0 |
| A. 200 characters, overlap 40 | 24 | 19 |
| B. 300 characters, overlap 40 | 15 | 10 |
| C. 200 characters, no overlap | 21 | 13 |

### Finding the right piece (15 customer questions)

| Way of cutting | Right piece in the top 3 | Right piece first |
|---|---|---|
| By section heading | 15 of 15 | 14 of 15 |
| A. 200, overlap 40 | 14 of 15 | 12 of 15 |
| B. 300, overlap 40 | 15 of 15 | 13 of 15 |
| C. 200, no overlap | 15 of 15 | 12 of 15 |

### The document without headings (`shipping-notes.md`, 4 questions)

| Way of cutting | Right piece in the top 3 | Right piece first |
|---|---|---|
| By section heading | 0 of 4 | 0 of 4 |
| A. 200, overlap 40 | 4 of 4 | 3 of 4 |
| B. 300, overlap 40 | 4 of 4 | 4 of 4 |
| C. 200, no overlap | 3 of 4 | 2 of 4 |

### Search findings

- **Cutting by section is strongest where it can be used.** It also hid a problem: `shipping-notes.md` has no headings, so it gave no pieces at all and could not be searched. Nothing warned about it.
- **Overlap protected a fact.** Without overlap, the rule "Orders above 100 dollars ship for free." was cut in two. Search returned a piece reading only "dollars ship for free", which could lead to a confident wrong answer that shipping is free with no minimum.
- **Adding a title clue to arbitrary pieces made things worse.** First-place results fell from 12 to 7 (A), 13 to 11 (B) and 12 to 7 (C), out of 15. A piece can start under one heading and run into the next, so its title can be wrong. Test a title clue before relying on it.
- **A perfect score can hide a weakness.** Version B scored 4 of 4 first on shipping only because the whole document is shorter than one piece.

The full guide, with a checklist for accepting a chunk, is in `guides/chunking-guide.md`.

### Two checks on the search itself

**Does the search model understand meaning? 5 of 6 pairs of sentences passed** (`embedding-check.ts`). It matched sentences with the same meaning in different words, and kept unrelated topics apart. It failed on opposites: "The payment went through successfully" and "The payment failed" were treated as similar, so a search for failed payments can return successful ones. The pass mark of 0.5 is a starting guess.

**What happens when the documents do not answer? 14 of 18 questions got the right treatment** (`no-answer-tests.ts`).

| Group | Handled correctly |
|---|---|
| Not covered: other topic (gift wrapping, support hours...) | 4 of 4 |
| Not related at all (capital of France, a cake recipe) | 2 of 2 |
| Answer is only implied (gift card, reset by phone...) | 3 of 3 |
| Control: documents do answer | 4 of 4 |
| Look-alike pair (wishlist against cart) | 1 of 2 |
| Sounds close, detail missing (fingerprint, PUT, internet off) | 0 of 3 |

All four failures are the questions that sound close to a covered topic. Search gives them a strong enough match that the system would present an unrelated piece as the answer. The closest unanswerable question (PUT requests, score 0.50) scored higher than the weakest answerable one ("sold out", 0.37), so **no single cut-off can separate the two kinds of question**. A stricter cut-off would refuse real questions, and a looser one lets unrelated text through. This is why the writer's own refusal is the safety net, and why the writer and the judge are tested below.

### Checking the writer and the judge

The writer was run on 10 questions (5 the documents answer, 5 they do not), using the section store and the top three pieces, at temperature 0. The judge then checked every answer against the pieces the writer had been given.

The judge is an AI too, so it was tested first on 4 planted answers whose correct verdict was known ("Up to 10 units" is supported, "Up to 20 units" is not, and so on). It got 4 of 4 right. It then gave identical verdicts on two full runs.

| Run | Pass | Warning | Fail |
|---|---|---|---|
| Main run (writer told how to refuse) | 6 | 3 | 1 |
| Scenario A: refusal instruction removed | 9 | 0 | 1 |
| Scenario B: writer given only the first piece | 0 | 1 | 0 |

**Reading the table**

- A **warning** means every fact was true, but after saying "the documents do not cover this" the writer added a sentence that was not about the question, for example answering a fingerprint question with "Registered customers must be able to sign in...". Three of the four such extras were judged misleading.
- The main run's one fail (a question about the internet going off during payment) is a **judge error on manual review**: none of the three pieces says anything about it, so the refusal was correct. Corrected, the main run is 7 pass, 3 warning, 0 fail. The judge's own verdicts are kept in `judge-results.json` for transparency.
- The fail in scenario A is real. The writer said "The documents do not mention fingerprint login, so it is not supported." The documents never say fingerprint login is unsupported.
- The writer's refusal rule is a trade-off, not a fix. With the rule, 3 misleading extras appeared. Without it, there were none, but one conclusion was made up. An untested idea is to keep the exact refusal sentence and remove only the permission to add a related fact.

## Limitations

- **`shipping-notes.md` is invisible to the section store.** It has no `##` headings, so cutting by section gives it no pieces. The fixed-size stores can search it. Adding headings is the cheapest fix.
- **The sold-out search miss is not fixed.** The wrong piece comes first for that question in every way of cutting. It is recorded as a finding, not solved.
- **Search cannot tell "not covered" from "covered".** The no-answer check failed on 4 of 18 questions, and no score cut-off separates them. The system depends on the writer to refuse.
- **The search model treats opposites as similar.** One of 6 sentence-pair checks failed.
- **Small samples.** 15 search questions, 4 shipping questions, 18 no-answer questions, 6 sentence pairs, 10 writer questions. The results show how the system behaves on these documents, not a general guarantee.
- **The writer was run on one model.** `openai/gpt-oss-20b` only. Other models were not compared.
- **The judge is an AI.** It passed its planted checks and was stable across two runs, but being stable is not the same as being right. One of its verdicts was wrong, and its "extra text" label was wrong once on an answer that was not a refusal. Read the reason line of every warning and fail yourself.
- **The check for "the writer admitted it" in scenario A uses a phrase list.** It missed "do not specify", and the judge corrected it. Phrase lists miss wordings.
- **Search depends on the embedding model.** The model reads about 256 word pieces. Text beyond that is ignored without an error, so very long pieces are a risk.

## How to run it

You need Node.js and a free [Groq](https://console.groq.com/) account (only the writer and judge scripts need the key).

1. Install the packages:

   ```
   npm install
   ```

2. Create a file called `.env` in the project folder. Never commit it. Put your key in it:

   ```
   GROQ_API_KEY=
   ```

3. Build the stores (the first run downloads a small embedding model, about 25 MB):

   ```
   npx tsx build-vector-store.ts
   npx tsx build-vector-store-fixed.ts
   ```

4. Run the checks:

   ```
   npx tsx embedding-check.ts
   npx tsx retrieval-tests.ts
   npx tsx compare-retrieval.ts
   npx tsx compare-titled.ts
   npx tsx shipping-retrieval-tests.ts
   npx tsx no-answer-tests.ts
   ```

5. Run the writer and the judge (these use Groq and wait between calls, so each takes a few minutes):

   ```
   npx tsx faithfulness-writer.ts
   npx tsx faithfulness-writer-no-refusal.ts
   npx tsx faithfulness-writer-top1.ts
   npx tsx faithfulness-judge.ts
   ```

   Run the writer scripts before the judge: the judge reads the answer files they save.

## What is in the repo

| File | What it does |
|---|---|
| `docs/` | The sample documents |
| `guides/chunking-guide.md` | How to cut documents, what we measured, and a step-by-step guide |
| `chunker.ts` | The shared code that cuts a document into pieces |
| `chunk-by-section.ts`, `chunk-fixed-size.ts` | Show the pieces each way of cutting produces |
| `build-vector-store.ts`, `build-vector-store-fixed.ts` | Build the searchable stores and check them |
| `check-shipping-split.ts` | Checks whether any key fact is split between two pieces |
| `embed-hello.ts`, `embedding-check.ts` | Smoke test and quality check of the embedding model |
| `search-store.ts` | Ask the store a question from the command line |
| `retrieval-tests.ts`, `compare-retrieval.ts`, `compare-titled.ts`, `shipping-retrieval-tests.ts` | Search tests and comparisons |
| `no-answer-tests.ts` | Checks that questions the documents cannot answer are not answered with unrelated text |
| `faithfulness-writer.ts` | The writer, with the main run and the answers saved |
| `faithfulness-writer-no-refusal.ts`, `faithfulness-writer-top1.ts` | Extra scenarios A and B |
| `faithfulness-judge.ts` | The judge, tested on planted answers, then used on all real answers |
| `*.json` | Saved stores, saved answers and `judge-results.json` |

## What is next

- Add headings to `shipping-notes.md`, or make the system warn when a document gives no pieces.
- Test the idea of keeping the exact refusal sentence while removing the permission to add a related fact.
- Compare a second writer model.