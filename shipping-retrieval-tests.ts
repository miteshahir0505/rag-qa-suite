// Load a built-in helper that reads files
import * as fs from "fs";

// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

// How many top results count as "found", and the model every store must use
const TOP_N = 3;
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";

// The document these questions are about
const DOCUMENT = "shipping-notes.md";

// The section store is checked only to show that it has no pieces from this document
const SECTION_STORE_FILE = "vector-store.json";

// The three fixed-size stores we compare
const storeSettings = [
  { label: "A: 200, overlap 40", file: "vector-store-fixed.json" },
  { label: "B: 300, overlap 40", file: "vector-store-fixed-300.json" },
  { label: "C: 200, no overlap", file: "vector-store-fixed-no-overlap.json" },
];

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

// Turns true/false into a green PASS or a red FAIL
function verdict(ok: boolean) {
  return ok ? paint(GREEN, "PASS") : paint(RED, "FAIL");
}

// Compares two lists of numbers and gives one score (higher = closer in meaning)
function similarity(a: number[], b: number[]) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total = total + a[i] * b[i];
  }
  return total;
}

// Makes all spaces and line breaks into single spaces, so texts can be compared fairly
function tidy(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

// Says in plain words where the right piece appeared
function describeRank(rank: number, total: number) {
  if (rank === 0) {
    return "no piece holds the whole sentence, so the answer cannot be found";
  }
  if (rank === 1) {
    return "the right piece was the FIRST result";
  }
  if (rank <= TOP_N) {
    return "the right piece was result number " + rank + " (another piece was shown first)";
  }
  return "the right piece was NOT in the top " + TOP_N + " (it was result number " + rank + " of " + total + ")";
}

// The questions. Each one is worded differently from the document on purpose.
// expectedSentence is the sentence of the document that answers it.
const tests = [
  {
    question: "How quickly will my order be sent out after I pay?",
    expectedSentence: "Orders are shipped within 2 working days of payment.",
  },
  {
    question: "How many days does normal delivery take?",
    expectedSentence: "Standard delivery takes 3 to 5 working days.",
  },
  {
    question: "How much extra do I pay for next-day delivery?",
    expectedSentence: "Express delivery takes 1 working day and costs 15 dollars extra.",
  },
  {
    question: "Do I have to pay for shipping on a big order?",
    expectedSentence: "Orders above 100 dollars ship for free.",
  },
];

// What we remember about one question on one store
type Outcome = {
  rank: number;
  ok: boolean;
};

async function main() {
  console.log(paint(CYAN, "SHIPPING QUESTIONS: CAN THE SEARCH FIND THE SHIPPING FACTS?"));
  console.log("Question being tested: " + DOCUMENT + " has no section headings, so the section strategy");
  console.log("cannot search it. Can the fixed-size stores find its facts when a customer asks?");
  console.log("");
  console.log(paint(CYAN, "HOW TO READ THIS REPORT"));
  console.log("  - Requirement: the piece that answers the question must be in the top " + TOP_N + " results.");
  console.log("  - A piece counts as right when it comes from " + DOCUMENT + " AND holds the whole answering sentence.");
  console.log("  - A sentence that was cut in half by the cutting cannot count, because nobody can read the full fact.");
  console.log("  - PASS = the right piece is in the top " + TOP_N + ". FAIL = it is not.");
  console.log("");

  // Check the section store first: does it hold any piece of this document?
  let sectionStorePieces = 0;
  if (fs.existsSync(SECTION_STORE_FILE)) {
    const sectionStore = JSON.parse(fs.readFileSync(SECTION_STORE_FILE, "utf8"));
    for (const entry of sectionStore.entries) {
      if (entry.source === DOCUMENT) {
        sectionStorePieces++;
      }
    }
  }
  console.log("Section store: " + sectionStorePieces + " pieces from " + DOCUMENT + ".");
  if (sectionStorePieces === 0) {
    console.log(paint(RED, "  Requirement: " + DOCUMENT + " must be searchable. Actual: it has no pieces, so every question below FAILS for the section store. Verdict: FAIL"));
  }
  console.log("");

  // Load every fixed-size store, and skip a store that is missing or built with another model
  const stores: any[] = [];
  for (const setting of storeSettings) {
    if (!fs.existsSync(setting.file)) {
      console.log(paint(YELLOW, "WARNING: " + setting.file + " was not found, so " + setting.label + " is skipped. Run: npx tsx build-vector-store-fixed.ts"));
      continue;
    }
    const data = JSON.parse(fs.readFileSync(setting.file, "utf8"));
    if (data.model !== MODEL_NAME) {
      console.log(paint(YELLOW, "WARNING: " + setting.file + " was built with another model, so " + setting.label + " is skipped."));
      continue;
    }
    // Count how many pieces of this document the store holds
    let ownPieces = 0;
    for (const entry of data.entries) {
      if (entry.source === DOCUMENT) {
        ownPieces++;
      }
    }
    stores.push({ label: setting.label, entries: data.entries });
    console.log("Loaded " + setting.label + ": " + data.entries.length + " pieces in total, " + ownPieces + " from " + DOCUMENT);
  }
  console.log("");

  // Load the model once and turn all questions into numbers in one go
  console.log("Loading the embedding model...");
  const embedder = await pipeline("feature-extraction", MODEL_NAME);
  const output = await embedder(tests.map((t) => t.question), { pooling: "mean", normalize: true });
  const questionVectors: number[][] = output.tolist();
  console.log(paint(GREEN, "Model loaded.") + "\n");

  // results[s][t] = the outcome of store s on test t
  const results: Outcome[][] = [];
  for (let s = 0; s < stores.length; s++) {
    results.push([]);
  }

  for (let t = 0; t < tests.length; t++) {
    const test = tests[t];
    console.log(paint(MAGENTA, "Test " + (t + 1) + " of " + tests.length + ": " + test.question));
    console.log("  Requirement: a piece holding the answer must be in the top " + TOP_N + " results");
    console.log("  Expected:    a piece from " + DOCUMENT + " that holds: \"" + test.expectedSentence + "\"");

    for (let s = 0; s < stores.length; s++) {
      const store = stores[s];

      // Score every piece against the question, then sort from best to worst
      const scored = store.entries.map((entry: any) => ({
        source: entry.source,
        text: entry.text,
        score: similarity(questionVectors[t], entry.vector),
      }));
      scored.sort((a: any, b: any) => b.score - a.score);

      // Find the first piece that counts as the right one
      let rank = 0;
      for (let r = 0; r < scored.length; r++) {
        if (scored[r].source === DOCUMENT && tidy(scored[r].text).includes(test.expectedSentence)) {
          rank = r + 1;
          break;
        }
      }

      const ok = rank >= 1 && rank <= TOP_N;
      results[s].push({ rank: rank, ok: ok });

      console.log("  " + store.label + ": " + describeRank(rank, scored.length) + " | " + verdict(ok));
      if (rank !== 1) {
        const firstText = tidy(scored[0].text);
        console.log(paint(YELLOW, "      First result instead: " + scored[0].source + " / " + firstText.slice(0, 70) + "..."));
      }
    }
    console.log("");
  }

  // THE SUMMARY, one line per store
  console.log(paint(CYAN, "SUMMARY"));
  console.log(paint(RED, "  Section store: 0 of " + tests.length + " found (" + DOCUMENT + " has no pieces)"));
  for (let s = 0; s < stores.length; s++) {
    let inTop = 0;
    let first = 0;
    for (const o of results[s]) {
      if (o.ok) {
        inTop++;
      }
      if (o.rank === 1) {
        first++;
      }
    }
    const lineColor = inTop === tests.length && first === tests.length ? GREEN : inTop === tests.length ? YELLOW : RED;
    console.log(paint(lineColor, "  " + stores[s].label + ": " + inTop + " of " + tests.length + " found in the top " + TOP_N + ", " + first + " of " + tests.length + " found FIRST"));
  }
  console.log("");

  console.log("Notes:");
  console.log("  - " + tests.length + " questions is a small sample. It shows how search behaves, not a guarantee.");
  console.log("  - Retrieval is plain math with no randomness, so running this again gives the same result.");
  console.log("  - A FAIL on a store where the sentence was split is a finding about the cutting, not about the search.");
}

// Run the function
main();