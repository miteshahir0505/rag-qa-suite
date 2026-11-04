// Load the secret key from the .env file
import "dotenv/config";

// Load the library that talks to the AI
import OpenAI from "openai";

// Load a built-in helper that reads and writes files
import * as fs from "fs";

// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

// The AI model that writes the answers. To try another model, change only this line.
const WRITER_MODEL = "openai/gpt-oss-20b";

// The saved store we search, how many pieces the writer gets, and the file for the answers
const STORE_FILE = "vector-store.json";
const TOP_N = 3;
const ANSWERS_FILE = "writer-answers.json";

// Milliseconds to rest between questions, so the free Groq tier is not flooded
const PAUSE_MS = 3000;

// How many times one call is tried again when Groq says "too many requests"
const MAX_TRIES = 6;

// The most words the writer may use up, including its hidden thinking.
// A reasoning model can spend its whole allowance thinking and leave an empty answer.
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

// The rules we give the writer. The exact refusal sentence makes a refusal easy to recognise.
const REFUSAL = "The documents do not cover this.";
const writerRules =
  "You are a customer support assistant. You are given numbered pieces of company documents and a question. " +
  "Answer using ONLY the information in those pieces. Do not use outside knowledge, and do not guess. " +
  "If the pieces do not contain the answer, reply with exactly this sentence: " + REFUSAL + " " +
  "You may add a related fact from the pieces after that sentence, but only if it is exactly what the pieces say. " +
  "Otherwise, answer in one or two plain sentences.";

// One fact the answer must contain. Any of the words in "accept" counts, but only as a whole word or number.
type Fact = { accept: string[] };

// One fact the answer may leave out. It is reported as a note, never as a failure.
type Detail = { label: string; accept: string[] };

// The questions. Rows 1 to 5 have an answer in the documents. Rows 6 to 10 do not.
// required = what the question itself asks for. omittable = related details we only note if they are missing.
type Test = {
  group: string;
  question: string;
  answerable: boolean;
  expected: string;
  required: Fact[];
  omittable: Detail[];
};

const tests: Test[] = [
  {
    group: "Answerable",
    question: "How many of the same product can I put in my basket?",
    answerable: true,
    expected: "10 units of the same item",
    required: [{ accept: ["10", "ten"] }],
    omittable: [],
  },
  {
    group: "Answerable",
    question: "How long do I have to wait after typing the wrong password too many times?",
    answerable: true,
    expected: "15 minutes",
    required: [{ accept: ["15 minutes", "fifteen minutes"] }],
    omittable: [{ label: "the lock starts after 5 failed sign-in attempts in a row", accept: ["5", "five"] }],
  },
  {
    group: "Answerable",
    question: "How long is the link to change my password valid?",
    answerable: true,
    expected: "1 hour",
    required: [{ accept: ["1 hour", "one hour", "60 minutes"] }],
    omittable: [{ label: "the link can be used only once", accept: ["once", "one time", "single use", "single-use"] }],
  },
  {
    group: "Answerable",
    question: "What happens if I add the same item again and go over the limit?",
    answerable: true,
    expected: "the quantity is set to 10 and a message explains the limit",
    required: [
      { accept: ["10", "ten"] },
      { accept: ["message", "explain", "explains", "explaining", "explained", "informed", "told", "notified", "notification"] },
    ],
    omittable: [],
  },
  {
    group: "Answerable, tricky",
    question: "What happens if I try to buy something that is sold out?",
    answerable: true,
    expected: "it cannot be added, and the button is replaced by an unavailable message",
    required: [{ accept: ["unavailable", "not available", "cannot", "can't", "can not", "not be added", "out of stock"] }],
    omittable: [],
  },
  {
    group: "Not covered, close to a topic",
    question: "Can I log in with my fingerprint?",
    answerable: false,
    expected: "the writer says the documents do not cover this",
    required: [],
    omittable: [],
  },
  {
    group: "Not covered, close to a topic",
    question: "Does the API support PUT requests to change an order?",
    answerable: false,
    expected: "the writer says the documents do not cover this",
    required: [],
    omittable: [],
  },
  {
    group: "Not covered, close to a topic",
    question: "If the internet goes off while I pay, will the order still be processed?",
    answerable: false,
    expected: "the writer says the documents do not cover this",
    required: [],
    omittable: [],
  },
  {
    group: "Not covered, close to a topic",
    question: "Does the wishlist hold more than 50 items?",
    answerable: false,
    expected: "the writer says the documents do not cover this",
    required: [],
    omittable: [],
  },
  {
    group: "Not covered, far away",
    question: "What is the capital of France?",
    answerable: false,
    expected: "the writer says the documents do not cover this",
    required: [],
    omittable: [],
  },
];

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

// Says whether a word or number appears in the text as a WHOLE word.
// This stops "5" from matching inside "15".
function containsWhole(text: string, word: string) {
  const safe = word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp("(^|[^a-z0-9])" + safe + "([^a-z0-9]|$)", "i");
  return pattern.test(text);
}

// Says whether any of the accepted words appears as a whole word
function containsAny(text: string, accept: string[]) {
  for (const word of accept) {
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
  console.log(paint(CYAN, "WRITER: DOES THE AI ANSWER FROM THE DOCUMENTS?"));
  console.log("Question being tested: given the pieces the search found, does the writer answer");
  console.log("only from them, and admit it when the documents do not cover the question?");
  console.log("");
  console.log(paint(CYAN, "HOW TO READ THIS REPORT"));
  console.log("  - Each question goes to the section store. The top " + TOP_N + " pieces are given to the writer (" + WRITER_MODEL + ").");
  console.log("  - Questions the documents answer: PASS = the answer contains what the question asks for.");
  console.log("    FAIL = something the question asks for is missing.");
  console.log("  - A correct answer that leaves out a related detail is still a PASS. The detail is shown as an");
  console.log("    'Omitted detail' note, so a reader can see it. It is not counted as a failure.");
  console.log("  - This script does NOT yet check for invented extras. The judge does that in the next step.");
  console.log("  - Questions the documents do not answer: LOOKS OK = the writer said the documents do not cover it.");
  console.log("    NEEDS JUDGE = the writer answered anyway. The judge decides whether that answer was invented.");
  console.log("  - If the writer adds a sentence after a refusal, it is shown as 'Extra fact after the refusal'.");
  console.log("    The judge checks that sentence, because it can still mislead a reader.");
  console.log("  - Even LOOKS OK is not final.");
  console.log("");

  // Load the saved store
  if (!fs.existsSync(STORE_FILE)) {
    console.log(paint(RED, STORE_FILE + " was not found. Run: npx tsx build-vector-store.ts"));
    return;
  }
  const store = JSON.parse(fs.readFileSync(STORE_FILE, "utf8"));
  console.log("Store loaded: " + store.entries.length + " pieces.");

  // Load the same model that built the store, and turn all questions into numbers
  console.log("Loading the embedding model...");
  const embedder = await pipeline("feature-extraction", store.model);
  const output = await embedder(tests.map((t) => t.question), { pooling: "mean", normalize: true });
  const questionVectors: number[][] = output.tolist();
  console.log(paint(GREEN, "Model loaded.") + "\n");

  // What we collect for the summary and for the judge
  const saved: any[] = [];
  let answerablePassed = 0;
  let answerableCounted = 0;
  let omittedCount = 0;
  let refused = 0;
  let refusedWithExtra = 0;
  let answeredAnyway = 0;
  let notCounted = 0;

  for (let t = 0; t < tests.length; t++) {
    const test = tests[t];

    // Find the top pieces for this question
    const scored = store.entries.map((entry: any) => ({
      source: entry.source,
      section: entry.section,
      text: entry.text,
      score: similarity(questionVectors[t], entry.vector),
    }));
    scored.sort((a: any, b: any) => b.score - a.score);
    const pieces = scored.slice(0, TOP_N);

    // Build the message for the writer: the pieces, then the question
    let prompt = "Pieces of the documents:\n\n";
    for (let p = 0; p < pieces.length; p++) {
      prompt += "Piece " + (p + 1) + " (" + pieces[p].source + ", " + pieces[p].section + "): " + pieces[p].text + "\n\n";
    }
    prompt += "Question: " + test.question;

    // Ask the writer
    const result = await askWriter(prompt);

    // Show the test like a line in a QA report
    console.log(paint(MAGENTA, "Test " + (t + 1) + " of " + tests.length + " [" + test.group + "]: " + test.question));
    if (test.answerable) {
      console.log("  Requirement: the answer must state what the question asks for, from the documents");
    } else {
      console.log("  Requirement: the writer must say the documents do not cover this, and must not invent an answer");
    }
    console.log("  Expected:    " + test.expected);
    console.log("  Pieces given to the writer:");
    for (let p = 0; p < pieces.length; p++) {
      console.log("    " + (p + 1) + ". " + pieces[p].source + " / " + pieces[p].section);
    }

    // Work out the verdict
    let verdictText = "";
    let omittedLabels: string[] = [];
    let extra = "";
    const answerLine = result.text.replace(/\s+/g, " ");
    if (result.status === "rate") {
      console.log("  Actual:      Groq kept refusing the call, so nothing was learned about this question.");
      verdictText = paint(YELLOW, "RATE LIMITED (not counted)");
      notCounted++;
    } else if (result.status === "empty") {
      console.log("  Actual:      the writer left an empty answer (it may have used its whole allowance thinking).");
      verdictText = test.answerable ? paint(RED, "FAIL") : paint(YELLOW, "NEEDS JUDGE");
      if (test.answerable) {
        answerableCounted++;
      } else {
        answeredAnyway++;
      }
    } else if (result.status === "error") {
      console.log("  Actual:      the AI call failed: " + answerLine);
      verdictText = paint(YELLOW, "ERROR (not counted)");
      notCounted++;
    } else {
      console.log("  Actual:      the writer answered: " + answerLine);
      if (test.answerable) {
        // Check what the question itself asks for
        const missing: string[] = [];
        for (const fact of test.required) {
          if (!containsAny(result.text, fact.accept)) {
            missing.push(fact.accept[0]);
          }
        }
        // Check the related details the answer may leave out
        for (const detail of test.omittable) {
          if (!containsAny(result.text, detail.accept)) {
            omittedLabels.push(detail.label);
          }
        }
        answerableCounted++;
        if (missing.length === 0) {
          answerablePassed++;
          verdictText = paint(GREEN, "PASS");
        } else {
          verdictText = paint(RED, "FAIL") + " (the answer does not contain: " + missing.join(", ") + ")";
        }
        if (omittedLabels.length > 0) {
          omittedCount++;
          console.log(paint(YELLOW, "  Omitted detail (not a failure): " + omittedLabels.join("; ")));
        }
      } else if (looksLikeRefusal(result.text)) {
        refused++;
        verdictText = paint(GREEN, "LOOKS OK");
        extra = extraAfterRefusal(result.text);
        if (extra !== "") {
          refusedWithExtra++;
          console.log(paint(YELLOW, "  Extra fact after the refusal (the judge will check it): " + extra));
        }
      } else {
        answeredAnyway++;
        verdictText = paint(YELLOW, "NEEDS JUDGE") + " (the writer answered instead of saying the documents do not cover it)";
      }
    }
    console.log("  Verdict:     " + verdictText);
    console.log("  Technical detail: best match score " + pieces[0].score.toFixed(2) + ", answer status " + result.status);
    console.log("");

    // Remember everything the judge will need
    saved.push({
      number: t + 1,
      group: test.group,
      question: test.question,
      answerable: test.answerable,
      expected: test.expected,
      pieces: pieces.map((p: any) => ({ source: p.source, section: p.section, text: p.text })),
      answer: result.text,
      status: result.status,
      omittedDetails: omittedLabels,
      extraAfterRefusal: extra,
    });

    // Short pause before the next question
    await wait(PAUSE_MS);
  }

  // Save the answers (each run replaces the old file)
  fs.writeFileSync(ANSWERS_FILE, JSON.stringify({ writerModel: WRITER_MODEL, answers: saved }, null, 2));

  // THE SUMMARY, in plain sentences
  const notCoveredTotal = tests.filter((t) => !t.answerable).length;
  console.log(paint(CYAN, "SUMMARY"));
  const answerColor = answerableCounted > 0 && answerablePassed === answerableCounted ? GREEN : RED;
  console.log(paint(answerColor, "  Questions the documents answer: " + answerablePassed + " of " + answerableCounted + " contained what the question asks for."));
  if (omittedCount > 0) {
    console.log(paint(YELLOW, "    " + omittedCount + " of those answers left out a related detail (shown above as 'Omitted detail')."));
  }
  console.log(paint(refused === notCoveredTotal ? GREEN : YELLOW, "  Questions the documents do not answer: " + refused + " of " + notCoveredTotal + " got a refusal, " + answeredAnyway + " got an answer anyway (the judge decides if it was invented)."));
  if (refusedWithExtra > 0) {
    console.log(paint(YELLOW, "    " + refusedWithExtra + " refusal(s) added an extra fact. The judge will check each one."));
  }
  if (notCounted > 0) {
    console.log(paint(YELLOW, "  " + notCounted + " question(s) were not counted because of rate limits or errors."));
  }
  console.log("  The writer's answers are saved in " + ANSWERS_FILE + " for the judge in the next step.");
  console.log("");
  console.log("Note: the writer is an AI and can answer differently each time, even with the same question.");
  console.log("Note: the requirement for tests 2 and 3 was changed after the first run, to ask only what the question asks.");
}

// Run the function
main();