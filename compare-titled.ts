// Load a built-in helper that reads files and folders
import * as fs from "fs";

// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

// Load the shared fixed-size cutting code
import { splitIntoFixedChunks } from "./chunker";

// The folder with our sample documents
const DOCS_FOLDER = "docs";

// How many top results count as "found", and the model every store must use
const TOP_N = 3;
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";

// The section store is the reference: it also tells us the text of each expected section
const SECTION_STORE_FILE = "vector-store.json";

// The three ways of cutting. Each one is tested twice: plain (saved store) and titled (built now).
const variants = [
  { letter: "A", size: 200, overlap: 40, plainFile: "vector-store-fixed.json" },
  { letter: "B", size: 300, overlap: 40, plainFile: "vector-store-fixed-300.json" },
  { letter: "C", size: 200, overlap: 0, plainFile: "vector-store-fixed-no-overlap.json" },
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

// Cuts a text into sentences
function splitSentences(text: string) {
  return tidy(text).split(/(?<=[.!?])\s+/);
}

// Turns a file name into a readable label, for example "add-to-cart-prd.md" becomes "add to cart prd"
function documentLabel(fileName: string) {
  return fileName.replace(".md", "").replace(/-/g, " ");
}

// Finds the nearest heading above a piece of text.
// If the document has no heading above it, the document title (a line starting with "# ") is used.
function findHeading(fullText: string, pieceText: string) {
  const clean = tidy(fullText);
  const position = clean.indexOf(tidy(pieceText));

  // If the piece cannot be located in the document, we cannot say which heading it sits under
  if (position === -1) {
    return { heading: "", found: false };
  }

  // Collect the document title and every "## " heading with its position in the tidy text
  const lines = fullText.split(/\r?\n/);
  let title = "";
  const headings: { name: string; position: number }[] = [];
  for (const line of lines) {
    if (line.startsWith("## ")) {
      const name = line.slice(3).trim();
      headings.push({ name: name, position: clean.indexOf("## " + name) });
    } else if (line.startsWith("# ") && title === "") {
      title = line.slice(2).trim();
    }
  }

  // The nearest heading is the last one that starts at or before the piece
  let nearest = title;
  for (const heading of headings) {
    if (heading.position !== -1 && heading.position <= position) {
      nearest = heading.name;
    }
  }
  return { heading: nearest, found: true };
}

// Says in plain words where the right piece appeared
function describeRank(rank: number, total: number) {
  if (rank === 0) {
    return "no piece holds a whole sentence of the right section";
  }
  if (rank === 1) {
    return "the right piece was the FIRST result";
  }
  if (rank <= TOP_N) {
    return "the right piece was result number " + rank + " (another piece was shown first)";
  }
  return "the right piece was NOT in the top " + TOP_N + " (it was result number " + rank + " of " + total + ")";
}

// The same 15 questions as retrieval-tests.ts, so the comparison is fair
const tests = [
  { group: "Normal questions", question: "How many of the same product can I put in my basket?", expectedSource: "add-to-cart-prd.md", expectedSection: "Quantity limits" },
  { group: "Normal questions", question: "What happens if I try to buy something that is sold out?", expectedSource: "add-to-cart-prd.md", expectedSection: "Out of stock items" },
  { group: "Normal questions", question: "How long do I have to wait after typing the wrong password too many times?", expectedSource: "login-prd.md", expectedSection: "Account lock" },
  { group: "Normal questions", question: "How long before an inactive user is logged out?", expectedSource: "login-prd.md", expectedSection: "Session timeout" },
  { group: "Normal questions", question: "Can I pay with cash when the parcel arrives?", expectedSource: "checkout-prd.md", expectedSection: "Payment methods" },
  { group: "Normal questions", question: "How do I get my money back for a returned order?", expectedSource: "checkout-prd.md", expectedSection: "Refunds" },
  { group: "Normal questions", question: "What does the API return when the order id is unknown?", expectedSource: "orders-api.md", expectedSection: "Get an order" },
  { group: "Normal questions", question: "What happens when I send more than 100 requests in one minute?", expectedSource: "orders-api.md", expectedSection: "Rate limits" },
  { group: "Spelling mistakes", question: "How meny items can I put in my baskit?", expectedSource: "add-to-cart-prd.md", expectedSection: "Quantity limits" },
  { group: "Spelling mistakes", question: "wat happens if my payment fales", expectedSource: "checkout-prd.md", expectedSection: "Failed payments" },
  { group: "Very short questions", question: "refund?", expectedSource: "checkout-prd.md", expectedSection: "Refunds" },
  { group: "Very short questions", question: "sold out", expectedSource: "add-to-cart-prd.md", expectedSection: "Out of stock items" },
  { group: "Two chunks could fit", question: "What happens if I add the same item again and go over the limit?", expectedSource: "add-to-cart-prd.md", expectedSection: "Duplication Of Items" },
  { group: "Similar chunks nearby", question: "How long is the link to change my password valid?", expectedSource: "login-prd.md", expectedSection: "Password reset" },
  { group: "Similar chunks nearby", question: "What is the biggest purchase I can make in one order?", expectedSource: "checkout-prd.md", expectedSection: "Order limits" },
];

// What we remember about one question on one store
type Outcome = {
  rank: number;
  ok: boolean;
  firstPlaceText: string;
};

async function main() {
  console.log(paint(CYAN, "FAIR TITLE TEST: DOES ADDING THE TITLE CLOSE THE GAP?"));
  console.log("Question being tested: the section store embeds each section title with its text, and the");
  console.log("fixed-size stores did not. If the fixed-size pieces also get a title clue, do they find answers");
  console.log("as well as the section store?");
  console.log("");
  console.log(paint(CYAN, "HOW TO READ THIS REPORT"));
  console.log("  - Seven stores are compared: the section store, three plain fixed-size stores (A, B, C),");
  console.log("    and three titled fixed-size stores (the same cuts, with a title clue added).");
  console.log("  - A titled piece is embedded as: document name + nearest heading + the piece text.");
  console.log("    For a document with no headings, the document title is used as the heading.");
  console.log("  - Requirement: the piece that answers the question must be in the top " + TOP_N + " results.");
  console.log("  - A piece counts as right when it comes from the expected file AND holds at least one whole");
  console.log("    sentence of the expected section. The title clue is not part of the text that is checked.");
  console.log("");

  // Load the section store first: it also tells us the text of every expected section
  if (!fs.existsSync(SECTION_STORE_FILE)) {
    console.log(paint(RED, SECTION_STORE_FILE + " was not found. Run: npx tsx build-vector-store.ts"));
    return;
  }
  const sectionStore = JSON.parse(fs.readFileSync(SECTION_STORE_FILE, "utf8"));

  // Every store goes into this list
  const stores: any[] = [];
  stores.push({ label: "Section store", bySection: true, kind: "section", letter: "", entries: sectionStore.entries });

  // Load the three plain stores that were built earlier
  for (const variant of variants) {
    if (!fs.existsSync(variant.plainFile)) {
      console.log(paint(RED, variant.plainFile + " was not found. Run: npx tsx build-vector-store-fixed.ts"));
      return;
    }
    const data = JSON.parse(fs.readFileSync(variant.plainFile, "utf8"));
    if (data.model !== MODEL_NAME) {
      console.log(paint(RED, variant.plainFile + " was built with another model, so the scores would mean nothing."));
      return;
    }
    stores.push({ label: variant.letter + " plain", bySection: false, kind: "plain", letter: variant.letter, entries: data.entries });
  }

  // Load the model once
  console.log("Loading the embedding model...");
  const embedder = await pipeline("feature-extraction", MODEL_NAME);
  console.log(paint(GREEN, "Model loaded.") + "\n");

  // Read every document once
  const fileNames = fs
    .readdirSync(DOCS_FOLDER)
    .filter((name) => name.endsWith(".md"))
    .sort();
  const texts: { [name: string]: string } = {};
  for (const fileName of fileNames) {
    texts[fileName] = fs.readFileSync(DOCS_FOLDER + "/" + fileName, "utf8");
  }

  // Build the three titled stores in memory
  console.log(paint(CYAN, "BUILDING THE TITLED STORES (in memory, nothing is saved)"));
  for (const variant of variants) {
    // Cut every document with this version's settings, and work out the title clue of each piece
    const pieces: any[] = [];
    let headingNotFound = 0;
    for (const fileName of fileNames) {
      const chunks = splitIntoFixedChunks(fileName, texts[fileName], variant.size, variant.overlap);
      for (const chunk of chunks) {
        const found = findHeading(texts[fileName], chunk.text);
        if (!found.found) {
          headingNotFound++;
        }
        const clue = documentLabel(fileName) + ". " + (found.heading !== "" ? found.heading + ". " : "");
        pieces.push({ source: chunk.source, text: chunk.text, textToEmbed: clue + chunk.text });
      }
    }

    // Turn every titled piece into numbers
    const output = await embedder(pieces.map((p) => p.textToEmbed), { pooling: "mean", normalize: true });
    const vectors: number[][] = output.tolist();
    const entries = pieces.map((p, i) => ({ source: p.source, section: "", text: p.text, vector: vectors[i] }));

    // QA check 1: the titled store must have exactly the same pieces as the plain store, or the comparison is unfair
    let plainCount = 0;
    for (const store of stores) {
      if (store.kind === "plain" && store.letter === variant.letter) {
        plainCount = store.entries.length;
      }
    }
    const sameCuts = entries.length === plainCount;
    console.log(paint(MAGENTA, variant.letter + " titled: ") + entries.length + " pieces built");
    console.log("  Check 1: same number of pieces as " + variant.letter + " plain | expected " + plainCount + ", actual " + entries.length + " | " + verdict(sameCuts));

    // QA check 2: every piece must have been given a heading
    console.log("  Check 2: every piece got a title clue | expected 0 pieces without one, actual " + headingNotFound + " | " + verdict(headingNotFound === 0));

    stores.push({ label: variant.letter + " titled", bySection: false, kind: "titled", letter: variant.letter, entries: entries });
  }
  console.log("");

  // For every test, find the sentences of the expected section (taken from the section store)
  const expectedSentences: string[][] = [];
  for (const test of tests) {
    let sectionText = "";
    for (const entry of sectionStore.entries) {
      if (entry.source === test.expectedSource && entry.section === test.expectedSection) {
        sectionText = entry.text;
      }
    }
    if (sectionText === "") {
      console.log(paint(RED, "Test data problem: section '" + test.expectedSection + "' was not found in " + test.expectedSource));
      return;
    }
    expectedSentences.push(splitSentences(sectionText));
  }

  // Turn all questions into numbers in one go
  const questionOutput = await embedder(tests.map((t) => t.question), { pooling: "mean", normalize: true });
  const questionVectors: number[][] = questionOutput.tolist();

  // results[s][t] = the outcome of store s on test t
  const results: Outcome[][] = [];
  for (let s = 0; s < stores.length; s++) {
    results.push([]);
  }

  for (let t = 0; t < tests.length; t++) {
    const test = tests[t];

    // Work out the result of every store for this question
    for (let s = 0; s < stores.length; s++) {
      const store = stores[s];

      // Score every piece against the question, then sort from best to worst
      const scored = store.entries.map((entry: any) => ({
        source: entry.source,
        section: entry.section,
        text: entry.text,
        score: similarity(questionVectors[t], entry.vector),
      }));
      scored.sort((a: any, b: any) => b.score - a.score);

      // Find the first piece that counts as the right one
      let rank = 0;
      for (let r = 0; r < scored.length; r++) {
        let isRight = false;
        if (store.bySection) {
          isRight = scored[r].source === test.expectedSource && scored[r].section === test.expectedSection;
        } else if (scored[r].source === test.expectedSource) {
          const pieceText = tidy(scored[r].text);
          for (const sentence of expectedSentences[t]) {
            if (pieceText.includes(sentence)) {
              isRight = true;
            }
          }
        }
        if (isRight) {
          rank = r + 1;
          break;
        }
      }

      results[s].push({
        rank: rank,
        ok: rank >= 1 && rank <= TOP_N,
        firstPlaceText: scored[0].source + " / " + tidy(scored[0].text).slice(0, 70),
      });
    }

    // Show the test like a line in a QA report
    console.log(paint(MAGENTA, "Test " + (t + 1) + " of " + tests.length + " [" + test.group + "]: " + test.question));
    console.log("  Requirement: the piece that answers this must be in the top " + TOP_N + " results");
    console.log("  Expected:    " + test.expectedSource + " / " + test.expectedSection);

    // If every store put the right piece first, one line is enough
    let allFirst = true;
    for (let s = 0; s < stores.length; s++) {
      if (results[s][t].rank !== 1) {
        allFirst = false;
      }
    }
    if (allFirst) {
      console.log("  All " + stores.length + " stores: the right piece was the FIRST result | " + verdict(true));
    } else {
      for (let s = 0; s < stores.length; s++) {
        const o = results[s][t];
        console.log("  " + stores[s].label + ": " + describeRank(o.rank, stores[s].entries.length) + " | " + verdict(o.ok));
        if (o.rank !== 1) {
          console.log(paint(YELLOW, "      First result instead: " + o.firstPlaceText + "..."));
        }
      }
    }
    console.log("");
  }

  // THE SUMMARY, one line per store
  console.log(paint(CYAN, "SUMMARY"));
  const topCounts: number[] = [];
  const firstCounts: number[] = [];
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
    topCounts.push(inTop);
    firstCounts.push(first);
    const lineColor = inTop === tests.length && first === tests.length ? GREEN : inTop === tests.length ? YELLOW : RED;
    console.log(paint(lineColor, "  " + stores[s].label + ": " + inTop + " of " + tests.length + " found in the top " + TOP_N + ", " + first + " of " + tests.length + " found FIRST"));
  }
  console.log("");

  // PLAIN VERSUS TITLED, for each way of cutting
  console.log(paint(CYAN, "DID THE TITLE CLUE HELP? (plain versus titled, same cuts)"));
  for (const variant of variants) {
    let plainIndex = -1;
    let titledIndex = -1;
    for (let s = 0; s < stores.length; s++) {
      if (stores[s].letter === variant.letter && stores[s].kind === "plain") {
        plainIndex = s;
      }
      if (stores[s].letter === variant.letter && stores[s].kind === "titled") {
        titledIndex = s;
      }
    }
    const change = firstCounts[titledIndex] - firstCounts[plainIndex];
    let words = "no change in first-place results";
    let lineColor = YELLOW;
    if (change > 0) {
      words = "better by " + change + " question(s)";
      lineColor = GREEN;
    } else if (change < 0) {
      words = "WORSE by " + Math.abs(change) + " question(s)";
      lineColor = RED;
    }
    console.log(paint(lineColor, "  " + variant.letter + ": found first " + firstCounts[plainIndex] + " -> " + firstCounts[titledIndex] + " of " + tests.length + " (" + words + "); in the top " + TOP_N + ": " + topCounts[plainIndex] + " -> " + topCounts[titledIndex]));
  }
  console.log("");

  // THE QUESTIONS WHERE A TITLED STORE GAVE A DIFFERENT POSITION FROM ITS PLAIN STORE
  console.log(paint(CYAN, "QUESTIONS WHERE THE TITLE CLUE CHANGED THE RESULT"));
  let changed = 0;
  for (let t = 0; t < tests.length; t++) {
    const lines: string[] = [];
    for (const variant of variants) {
      let plainIndex = -1;
      let titledIndex = -1;
      for (let s = 0; s < stores.length; s++) {
        if (stores[s].letter === variant.letter && stores[s].kind === "plain") {
          plainIndex = s;
        }
        if (stores[s].letter === variant.letter && stores[s].kind === "titled") {
          titledIndex = s;
        }
      }
      if (results[plainIndex][t].rank !== results[titledIndex][t].rank) {
        const before = results[plainIndex][t].rank === 0 ? "not found" : "result " + results[plainIndex][t].rank;
        const after = results[titledIndex][t].rank === 0 ? "not found" : "result " + results[titledIndex][t].rank;
        lines.push("      " + variant.letter + ": " + before + " -> " + after);
      }
    }
    if (lines.length > 0) {
      changed++;
      console.log("  - \"" + tests[t].question + "\"");
      for (const line of lines) {
        console.log(line);
      }
    }
  }
  if (changed === 0) {
    console.log("  None. The title clue did not change any position.");
  }
  console.log("");

  console.log("Notes:");
  console.log("  - " + tests.length + " questions is a small sample. It shows how search behaves, not a guarantee.");
  console.log("  - Retrieval is plain math with no randomness, so running this again gives the same result.");
  console.log("  - The title clue is built by code from the document, so a real project must test that code as well.");
  console.log("  - These questions are about four documents. shipping-notes.md is not asked about here (see shipping-retrieval-tests.ts).");
}

// Run the function
main();