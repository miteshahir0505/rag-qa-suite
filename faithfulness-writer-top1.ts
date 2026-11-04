// Load the secret key from the .env file
import "dotenv/config";

// Load the library that talks to the AI
import OpenAI from "openai";

// Load a built-in helper that reads and writes files
import * as fs from "fs";

// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

// The AI model that writes the answers. Same as the baseline run.
const WRITER_MODEL = "openai/gpt-oss-20b";

// The saved store we search. THE CHANGE FROM THE BASELINE: the writer gets only 1 piece, not 3.
const STORE_FILE = "vector-store.json";
const TOP_N = 1;

// The baseline answers (read only, never changed) and the file for this run's answer
const BASELINE_FILE = "writer-answers.json";
const ANSWERS_FILE = "writer-answers-top1.json";

// How many times one call is tried again when Groq says "too many requests"
const MAX_TRIES = 6;

// The most words the writer may use up, including its hidden thinking
const MAX_OUTPUT = 2000;

// Set up the connection to Groq, using your key.
// maxRetries 0 switches off the library's hidden retries, so only our own retry logic runs.
const client = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
  maxRetries: 0,
});

// COLORS: standard terminal color codes, used only on a real terminal screen
const USE_COLOR = process.stdout.isTTY === true && !process.env.NO_COLOR;
const YELLOW = "33";
const GREEN = "32";
const RED = "31";
const CYAN = "1;36";
const MAGENTA = "1;35";

// Wraps a piece of text in a color (or leaves it plain if colors are off)
function paint(colorCode: string, text: string) {
  if (!USE_COLOR) {
    return text;
  }
  return "\u001b[" + colorCode + "m" + text + "\u001b[0m";
}

// The writer rules are EXACTLY the baseline rules, including the refusal instruction.
const REFUSAL = "The documents do not cover this.";
const writerRules =
  "You are a customer support assistant. You are given numbered pieces of company documents and a question. " +
  "Answer using ONLY the information in those pieces. Do not use outside knowledge, and do not guess. " +
  "If the pieces do not contain the answer, reply with exactly this sentence: " + REFUSAL + " " +
  "You may add a related fact from the pieces after that sentence, but only if it is exactly what the pieces say. " +
  "Otherwise, answer in one or two plain sentences.";

// The one question of this scenario, and the words that would show the writer knows the right answer.
// The right answer is NOT in the piece the writer will get, so those words would mean the answer came from somewhere else.
const QUESTION = "What happens if I try to buy something that is sold out?";
const EXPECTED = "the writer says the documents do not cover this (the right piece is not among the pieces given)";
const RIGHT_ANSWER_WORDS = ["unavailable", "not available", "cannot be added", "can't be added", "out of stock", "sold out"];

// Waits for a short time
function wait(milliseconds: number) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

// Compares two lists of numbers and gives one score (higher = closer in meaning)
function similarity(a: number[], b: number[]) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total = total + a[i] * b[i];
  }
  return total;
}

// Works out how many seconds to wait after a "too many requests" error
function secondsToWait(error: any, attempt: number) {
  let hint: any = null;
  const headers = error.headers;
  if (headers) {
    hint = typeof headers.get === "function" ? headers.get("retry-after") : headers["retry-after"];
  }
  let seconds = Number(hint);
  if (!seconds || seconds < 0) {
    seconds = attempt * 15;
  }
  return Math.min(seconds + 2, 90);
}

// Asks the writer. Returns the answer and a status:
//   ok = a real answer, empty = the writer left no answer, rate = Groq kept refusing, error = another problem
async function askWriter(prompt: string) {
  for (let attempt = 1; attempt <= MAX_TRIES; attempt++) {
    try {
      const response = await client.chat.completions.create({
        model: WRITER_MODEL,
        messages: [
          { role: "system", content: writerRules },
          { role: "user", content: prompt },
        ],
        temperature: 0,
        max_tokens: MAX_OUTPUT,
      });
      const text = (response.choices[0].message.content || "").trim();
      if (text === "") {
        return { text: "", status: "empty" };
      }
      return { text: text, status: "ok" };
    } catch (error: any) {
      if (error.status === 429 && attempt < MAX_TRIES) {
        const seconds = secondsToWait(error, attempt);
        console.log("    (rate limit reached, waiting " + seconds + " seconds, try " + attempt + " of " + MAX_TRIES + ")");
        await wait(seconds * 1000);
      } else if (error.status === 429) {
        return { text: "", status: "rate" };
      } else {
        return { text: String(error.message).slice(0, 150), status: "error" };
      }
    }
  }
  return { text: "", status: "rate" };
}

// Says whether a word or phrase appears in the text as a WHOLE word
function containsWhole(text: string, word: string) {
  const safe = word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp("(^|[^a-z0-9])" + safe + "([^a-z0-9]|$)", "i");
  return pattern.test(text);
}

// Says whether any of the words appears as a whole word
function containsAny(text: string, words: string[]) {
  for (const word of words) {
    if (containsWhole(text, word)) {
      return true;
    }
  }
  return false;
}

// Says whether the writer refused, in any of the usual ways
function looksLikeRefusal(answer: string) {
  const lower = answer.toLowerCase();
  return (
    lower.includes("do not cover") ||
    lower.includes("does not cover") ||
    lower.includes("not covered") ||
    lower.includes("do not contain") ||
    lower.includes("does not contain")
  );
}

// If the writer refused and then added more text, returns the added text (otherwise an empty text)
function extraAfterRefusal(answer: string) {
  let rest = answer.replace(/\s+/g, " ");
  rest = rest.replace(REFUSAL, "");
  return rest.trim();
}

async function main() {
  console.log(paint(CYAN, "EXTRA SCENARIO B: WHAT IF THE WRITER GETS ONLY THE FIRST PIECE?"));
  console.log("Question being tested: for the sold-out question, the search puts the wrong piece first.");
  console.log("If the writer is given ONLY that first piece, does it admit the documents given do not answer,");
  console.log("or does it answer from the wrong piece?");
  console.log("");
  console.log(paint(CYAN, "HOW TO READ THIS REPORT"));
  console.log("  - Same writer (" + WRITER_MODEL + "), same rules, same temperature as the baseline. Only the number of pieces changes: " + TOP_N + " instead of 3.");
  console.log("  - The first piece for this question is about failed payments. The piece that really answers it (out of stock) is not given.");
  console.log("  - LOOKS OK = the writer says the documents do not cover this.");
  console.log("  - FAIL = the writer states the sold-out behaviour even though the piece given does not contain it.");
  console.log("  - NEEDS JUDGE = the writer gave some other answer. The judge decides if it was fair or misleading.");
  console.log("");

  // Load the saved store
  if (!fs.existsSync(STORE_FILE)) {
    console.log(paint(RED, STORE_FILE + " was not found. Run: npx tsx build-vector-store.ts"));
    return;
  }
  const store = JSON.parse(fs.readFileSync(STORE_FILE, "utf8"));
  console.log("Store loaded: " + store.entries.length + " pieces.");

  // Load the same model that built the store, and turn the question into numbers
  console.log("Loading the embedding model...");
  const embedder = await pipeline("feature-extraction", store.model);
  const output = await embedder([QUESTION], { pooling: "mean", normalize: true });
  const questionVector: number[] = output.tolist()[0];
  console.log(paint(GREEN, "Model loaded.") + "\n");

  // Find the top pieces for this question
  const scored = store.entries.map((entry: any) => ({
    source: entry.source,
    section: entry.section,
    text: entry.text,
    score: similarity(questionVector, entry.vector),
  }));
  scored.sort((a: any, b: any) => b.score - a.score);
  const pieces = scored.slice(0, TOP_N);

  // Build the message for the writer: the piece, then the question
  let prompt = "Pieces of the documents:\n\n";
  for (let p = 0; p < pieces.length; p++) {
    prompt += "Piece " + (p + 1) + " (" + pieces[p].source + ", " + pieces[p].section + "): " + pieces[p].text + "\n\n";
  }
  prompt += "Question: " + QUESTION;

  // Ask the writer
  const result = await askWriter(prompt);
  const answerLine = result.text.replace(/\s+/g, " ");

  // Show the test like a line in a QA report
  console.log(paint(MAGENTA, "Test 1 of 1 [Answerable, but the right piece is not given]: " + QUESTION));
  console.log("  Requirement: the writer must not answer from a piece that does not contain the answer");
  console.log("  Expected:    " + EXPECTED);
  console.log("  Piece given to the writer:");
  console.log("    1. " + pieces[0].source + " / " + pieces[0].section);
  console.log("       Text: " + pieces[0].text);

  // Work out the verdict
  let verdictText = "";
  let extra = "";
  if (result.status === "rate") {
    console.log("  Actual:      Groq kept refusing the call, so nothing was learned.");
    verdictText = paint(YELLOW, "RATE LIMITED (not counted)");
  } else if (result.status === "error") {
    console.log("  Actual:      the AI call failed: " + answerLine);
    verdictText = paint(YELLOW, "ERROR (not counted)");
  } else if (result.status === "empty") {
    console.log("  Actual:      the writer left an empty answer (it may have used its whole allowance thinking).");
    verdictText = paint(YELLOW, "NEEDS JUDGE");
  } else {
    console.log("  Actual:      the writer answered: " + answerLine);
    if (looksLikeRefusal(result.text)) {
      extra = extraAfterRefusal(result.text);
      verdictText = paint(GREEN, "LOOKS OK");
      if (extra !== "") {
        console.log(paint(YELLOW, "  Extra fact after the refusal (the judge will check it): " + extra));
      }
    } else if (containsAny(result.text, RIGHT_ANSWER_WORDS)) {
      verdictText = paint(RED, "FAIL") + " (the answer describes sold-out behaviour that is not in the piece given)";
    } else {
      verdictText = paint(YELLOW, "NEEDS JUDGE") + " (the writer answered instead of saying the documents do not cover it)";
    }
  }
  console.log("  Verdict:     " + verdictText);
  console.log("  Technical detail: best match score " + pieces[0].score.toFixed(2) + ", answer status " + result.status);
  console.log("");

  // Save the answer (each run replaces the old file)
  fs.writeFileSync(
    ANSWERS_FILE,
    JSON.stringify(
      {
        writerModel: WRITER_MODEL,
        scenario: "only the first piece given to the writer",
        question: QUESTION,
        pieces: pieces.map((p: any) => ({ source: p.source, section: p.section, text: p.text })),
        answer: result.text,
        status: result.status,
        extraAfterRefusal: extra,
      },
      null,
      2
    )
  );

  // THE COMPARISON with the baseline (top 3), read from the saved baseline answers
  console.log(paint(CYAN, "COMPARISON WITH THE BASELINE (3 pieces given)"));
  if (fs.existsSync(BASELINE_FILE)) {
    const baseline = JSON.parse(fs.readFileSync(BASELINE_FILE, "utf8"));
    let found = false;
    for (const item of baseline.answers) {
      if (item.question === QUESTION) {
        found = true;
        console.log("  Baseline pieces given:");
        for (let p = 0; p < item.pieces.length; p++) {
          console.log("    " + (p + 1) + ". " + item.pieces[p].source + " / " + item.pieces[p].section);
        }
        console.log("  Baseline answer: " + item.answer.replace(/\s+/g, " "));
      }
    }
    if (!found) {
      console.log(paint(YELLOW, "  The sold-out question was not found in " + BASELINE_FILE + "."));
    }
  } else {
    console.log(paint(YELLOW, "  " + BASELINE_FILE + " was not found, so there is nothing to compare with."));
  }
  console.log("  This run:        " + (answerLine === "" ? "(no answer)" : answerLine));
  console.log("  This run's answer is saved in " + ANSWERS_FILE + ". The baseline file is unchanged.");
  console.log("");
  console.log("Note: one question and one run is a first look, not a verdict.");
  console.log("Note: the lesson is about the pipeline. When search puts the wrong piece first, the writer can only be as good as what it is given.");
}

// Run the function
main();