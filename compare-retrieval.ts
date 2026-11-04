// Load a built-in helper that reads files
import * as fs from "fs";

// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

// How many top results count as "found", and the model every store must use
const TOP_N = 3;
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";

// The section store is the reference: it also tells us the text of each expected section
const SECTION_STORE_FILE = "vector-store.json";

// The four stores we compare, side by side
const storeSettings = [
  { label: "Section store", file: "vector-store.json", bySection: true },
  { label: "A: 200, overlap 40", file: "vector-store-fixed.json", bySection: false },
  { label: "B: 300, overlap 40", file: "vector-store-fixed-300.json", bySection: false },
  { label: "C: 200, no overlap", file: "vector-store-fixed-no-overlap.json", bySection: false },
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

// Says in plain words where the right piece appeared
function describeRank(rank: number, total: number) {
  if (rank === 0) {
    return "the right piece does not exist in this store";
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
  console.log(paint(CYAN, "SIDE-BY-SIDE RETRIEVAL COMPARISON"));
  console.log("Question being tested: for the same customer questions, which way of cutting the");
  console.log("documents lets the search find the right piece most often?");
  console.log("");
  console.log(paint(CYAN, "HOW TO READ THIS REPORT"));
  console.log("  - Four stores are compared: the section store and three fixed-size versions (A, B, C).");
  console.log("  - Requirement: the piece that answers the question must be in the top " + TOP_N + " results.");
  console.log("  - PASS = it is in the top " + TOP_N + ". FAIL = it is not. A first-place result is better than a third-place one.");
  console.log("  - In the section store, 'the right piece' is the section named in the test.");
  console.log("  - In a fixed-size store there are no section names. A piece counts as right when it comes");
  console.log("    from the expected file AND holds at least one whole sentence of the expected section.");
  console.log(paint(YELLOW, "  - Fairness note: the section store embeds the section title together with the text."));
  console.log(paint(YELLOW, "    The fixed-size stores embed the text alone, so the section store has an extra clue."));
  console.log("");

  // Load the section store first: it also tells us the text of every expected section
  if (!fs.existsSync(SECTION_STORE_FILE)) {
    console.log(paint(RED, SECTION_STORE_FILE + " was not found. Run: npx tsx build-vector-store.ts"));
    return;
  }
  const sectionStore = JSON.parse(fs.readFileSync(SECTION_STORE_FILE, "utf8"));

  // Load every store, and skip a store that is missing or built with another model
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
    stores.push({ label: setting.label, bySection: setting.bySection, entries: data.entries });
    console.log("Loaded " + setting.label + ": " + data.entries.length + " pieces");
  }
  console.log("");

  // For every test, find the text of the expected section (we need it for the fixed-size stores)
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
    console.log(paint(MAGENTA, "Test " + (t + 1) + " of " + tests.length + " [" + test.group + "]: " + test.question));
    console.log("  Requirement: the piece that answers this must be in the top " + TOP_N + " results");
    console.log("  Expected:    " + test.expectedSource + " / " + test.expectedSection);

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
          // Section store: the source and section must match exactly
          isRight = scored[r].source === test.expectedSource && scored[r].section === test.expectedSection;
        } else if (scored[r].source === test.expectedSource) {
          // Fixed-size store: the piece must hold at least one whole sentence of the expected section
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

      const ok = rank >= 1 && rank <= TOP_N;
      const firstText = tidy(scored[0].text);
      results[s].push({
        rank: rank,
        ok: ok,
        firstPlaceText: scored[0].source + ": " + firstText.slice(0, 70),
      });

      // Show this store's result like a line in a QA report
      console.log("  " + store.label + ": " + describeRank(rank, scored.length) + " | " + verdict(ok));
      if (rank > 1) {
        console.log(paint(YELLOW, "      First result instead: " + scored[0].source + " / " + firstText.slice(0, 70) + "..."));
      }
    }
    console.log("");
  }

  // THE SUMMARY, one line per store
  console.log(paint(CYAN, "SUMMARY"));
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

  // THE RESULTS PER GROUP, so you can see where each way of cutting is strong or weak
  console.log(paint(CYAN, "RESULTS PER GROUP (found in the top " + TOP_N + " / found first)"));
  const groupNames: string[] = [];
  for (const test of tests) {
    if (!groupNames.includes(test.group)) {
      groupNames.push(test.group);
    }
  }
  for (const name of groupNames) {
    console.log("  " + name);
    for (let s = 0; s < stores.length; s++) {
      let groupTotal = 0;
      let groupTop = 0;
      let groupFirst = 0;
      for (let t = 0; t < tests.length; t++) {
        if (tests[t].group === name) {
          groupTotal++;
          if (results[s][t].ok) {
            groupTop++;
          }
          if (results[s][t].rank === 1) {
            groupFirst++;
          }
        }
      }
      const lineColor = groupTop === groupTotal && groupFirst === groupTotal ? GREEN : groupTop === groupTotal ? YELLOW : RED;
      console.log(paint(lineColor, "    " + stores[s].label + ": " + groupTop + " of " + groupTotal + " / " + groupFirst + " of " + groupTotal));
    }
  }
  console.log("");

  // THE QUESTIONS WHERE THE STORES DISAGREE (a real finding)
  console.log(paint(CYAN, "QUESTIONS WHERE THE STORES GAVE DIFFERENT RESULTS"));
  let disagreements = 0;
  for (let t = 0; t < tests.length; t++) {
    let allSame = true;
    for (let s = 1; s < stores.length; s++) {
      if (results[s][t].rank !== results[0][t].rank) {
        allSame = false;
      }
    }
    if (!allSame) {
      disagreements++;
      console.log("  - \"" + tests[t].question + "\"");
      for (let s = 0; s < stores.length; s++) {
        console.log("      " + stores[s].label + ": " + (results[s][t].rank === 0 ? "not found" : "result number " + results[s][t].rank));
      }
    }
  }
  if (disagreements === 0) {
    console.log(paint(GREEN, "  None. All stores gave the same position for every question."));
  }
  console.log("");

  console.log("Notes:");
  console.log("  - " + tests.length + " questions is a small sample. It shows how search behaves, not a guarantee.");
  console.log("  - The section store has the title clue and the fixed-size stores do not, so a section-store win is not a pure cutting win.");
  console.log("  - Fixed-size pieces overlap, so two neighbouring pieces can both be right. The report uses the first one.");
}

// Run the function
main();