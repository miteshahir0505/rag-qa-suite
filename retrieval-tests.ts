// Load a built-in helper that reads files
import * as fs from "fs";

// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

// The saved store we are testing, and how many top results count as "found"
const STORE_FILE = "vector-store.json";
const TOP_N = 3;

// COLORS: these are standard terminal color codes.
// Colors are used only when the output goes to a real terminal screen.
// If the output is saved to a file, the text stays plain.
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

// Compares two lists of numbers and gives one score (0 = nothing in common, 1 = same meaning)
function similarity(a: number[], b: number[]) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total = total + a[i] * b[i];
  }
  return total;
}

// Says in plain words where the right chunk appeared
function describeRank(rank: number, total: number) {
  if (rank === 0) {
    return "the expected chunk does not exist in the store (check the test data)";
  }
  if (rank === 1) {
    return "the right chunk was the FIRST result (the best possible outcome)";
  }
  return "the right chunk was result number " + rank + " of " + total + " (another chunk was shown before it)";
}

// The test cases. Each one has a group, a customer question, and the chunk that should answer it.
// The questions use different words from the documents on purpose.
const tests = [
  // GROUP 1: normal questions
  {
    group: "Normal questions",
    question: "How many of the same product can I put in my basket?",
    expectedSource: "add-to-cart-prd.md",
    expectedSection: "Quantity limits",
  },
  {
    group: "Normal questions",
    question: "What happens if I try to buy something that is sold out?",
    expectedSource: "add-to-cart-prd.md",
    expectedSection: "Out of stock items",
  },
  {
    group: "Normal questions",
    question: "How long do I have to wait after typing the wrong password too many times?",
    expectedSource: "login-prd.md",
    expectedSection: "Account lock",
  },
  {
    group: "Normal questions",
    question: "How long before an inactive user is logged out?",
    expectedSource: "login-prd.md",
    expectedSection: "Session timeout",
  },
  {
    group: "Normal questions",
    question: "Can I pay with cash when the parcel arrives?",
    expectedSource: "checkout-prd.md",
    expectedSection: "Payment methods",
  },
  {
    group: "Normal questions",
    question: "How do I get my money back for a returned order?",
    expectedSource: "checkout-prd.md",
    expectedSection: "Refunds",
  },
  {
    group: "Normal questions",
    question: "What does the API return when the order id is unknown?",
    expectedSource: "orders-api.md",
    expectedSection: "Get an order",
  },
  {
    group: "Normal questions",
    question: "What happens when I send more than 100 requests in one minute?",
    expectedSource: "orders-api.md",
    expectedSection: "Rate limits",
  },

  // GROUP 2: spelling mistakes (real users make them)
  {
    group: "Spelling mistakes",
    question: "How meny items can I put in my baskit?",
    expectedSource: "add-to-cart-prd.md",
    expectedSection: "Quantity limits",
  },
  {
    group: "Spelling mistakes",
    question: "wat happens if my payment fales",
    expectedSource: "checkout-prd.md",
    expectedSection: "Failed payments",
  },

  // GROUP 3: very short questions (little information to match on)
  {
    group: "Very short questions",
    question: "refund?",
    expectedSource: "checkout-prd.md",
    expectedSection: "Refunds",
  },
  {
    group: "Very short questions",
    question: "sold out",
    expectedSource: "add-to-cart-prd.md",
    expectedSection: "Out of stock items",
  },

  // GROUP 4: two chunks could fit (the expected one is the best answer, but a neighbour is also related)
  {
    group: "Two chunks could fit",
    question: "What happens if I add the same item again and go over the limit?",
    expectedSource: "add-to-cart-prd.md",
    expectedSection: "Duplication Of Items",
  },

  // GROUP 5: similar chunks sit close together (same document, same topic)
  {
    group: "Similar chunks nearby",
    question: "How long is the link to change my password valid?",
    expectedSource: "login-prd.md",
    expectedSection: "Password reset",
  },
  {
    group: "Similar chunks nearby",
    question: "What is the biggest purchase I can make in one order?",
    expectedSource: "checkout-prd.md",
    expectedSection: "Order limits",
  },
];

// One finished test, kept so the summary can group and explain the results
type Outcome = {
  group: string;
  question: string;
  expectedName: string;
  rank: number;
  ok: boolean;
  firstPlaceName: string;
};

async function main() {
  // Explain the check in plain words for anyone reading the output
  console.log(paint(CYAN, "RETRIEVAL QUALITY CHECK"));
  console.log("Question being tested: when a customer asks something, does the search");
  console.log("find the piece of the documents (a chunk) that really answers it?");
  console.log("");
  console.log(paint(CYAN, "HOW TO READ THIS REPORT"));
  console.log("  - For each question, the search ranks all chunks from best match to worst match.");
  console.log("  - Rank 1 = the first result. Rank 2 = the second result. And so on.");
  console.log("  - Score = how close in meaning the chunk is to the question (0 to 1, higher is closer).");
  console.log("    A high score does NOT prove the chunk answers the question.");
  console.log("  - Requirement: the right chunk must be among the top " + TOP_N + " results.");
  console.log("  - PASS = the right chunk is in the top " + TOP_N + ". FAIL = it is not.");
  console.log("  - A PASS is better when the right chunk is first, because a chatbot may only use the first result.");
  console.log("");

  // Load the saved store
  const store = JSON.parse(fs.readFileSync(STORE_FILE, "utf8"));
  const entries = store.entries;
  console.log("Store loaded: " + entries.length + " chunks.");

  // Load the same model that built the store
  console.log("Loading the embedding model...");
  const embedder = await pipeline("feature-extraction", store.model);
  console.log(paint(GREEN, "Model loaded.") + "\n");

  // All finished tests go here
  const outcomes: Outcome[] = [];

  for (let i = 0; i < tests.length; i++) {
    const test = tests[i];
    const expectedName = test.expectedSource + " / " + test.expectedSection;

    // Turn the question into numbers
    const output = await embedder([test.question], { pooling: "mean", normalize: true });
    const questionVector: number[] = output.tolist()[0];

    // Score every chunk, then sort from best to worst
    const scored = entries.map((entry: any) => ({
      source: entry.source,
      section: entry.section,
      score: similarity(questionVector, entry.vector),
    }));
    scored.sort((a: any, b: any) => b.score - a.score);

    // Find the rank of the expected chunk (1 = best). 0 means it is not in the store at all.
    let rank = 0;
    for (let r = 0; r < scored.length; r++) {
      if (scored[r].source === test.expectedSource && scored[r].section === test.expectedSection) {
        rank = r + 1;
      }
    }

    // The verdict: the expected chunk must be in the top N
    const ok = rank >= 1 && rank <= TOP_N;

    // Remember this test for the summary
    outcomes.push({
      group: test.group,
      question: test.question,
      expectedName: expectedName,
      rank: rank,
      ok: ok,
      firstPlaceName: scored[0].source + " / " + scored[0].section,
    });

    // Show the test like a line in a QA report
    console.log(paint(MAGENTA, "Test " + (i + 1) + " of " + tests.length + " [" + test.group + "]: " + test.question));
    console.log("  Requirement: the chunk that answers this must be in the top " + TOP_N + " results");
    console.log("  Expected:    " + expectedName);
    console.log("  Actual:      " + describeRank(rank, scored.length));
    console.log("  Top " + TOP_N + " returned:");
    for (let r = 0; r < TOP_N; r++) {
      const isExpected = scored[r].source === test.expectedSource && scored[r].section === test.expectedSection;
      const mark = isExpected ? "  <-- the right chunk" : "";
      console.log("    " + (r + 1) + ". " + scored[r].source + " / " + scored[r].section + " (score " + scored[r].score.toFixed(2) + ")" + mark);
    }
    if (ok && rank > 1) {
      console.log(paint(YELLOW, "  Note: passed, but the first result was a different chunk."));
    }
    console.log("  Verdict:     " + (ok ? paint(GREEN, "PASS") : paint(RED, "FAIL")));
    console.log("");
  }

  // Count the totals
  let inTopN = 0;
  let atRank1 = 0;
  for (const o of outcomes) {
    if (o.ok) {
      inTopN++;
    }
    if (o.rank === 1) {
      atRank1++;
    }
  }
  const total = outcomes.length;

  // THE SUMMARY, written as plain sentences
  console.log(paint(CYAN, "SUMMARY"));
  const topColor = inTopN === total ? GREEN : RED;
  console.log(paint(topColor, "  " + inTopN + " of " + total + " questions PASSED: the right chunk was in the top " + TOP_N + " results."));
  if (inTopN < total) {
    console.log(paint(RED, "  " + (total - inTopN) + " question(s) FAILED: the right chunk was not in the top " + TOP_N + "."));
  }
  const firstColor = atRank1 === total ? GREEN : YELLOW;
  console.log(paint(firstColor, "  " + atRank1 + " of " + total + " questions had the right chunk as the FIRST result."));
  if (atRank1 < total) {
    console.log("  In the other " + (total - atRank1) + ", a different chunk was shown first. A chatbot that uses only the first result would answer those wrongly.");
  }
  console.log("");

  // The results per group, so you can see WHERE search is strong or weak
  console.log(paint(CYAN, "RESULTS PER GROUP"));
  const groupNames: string[] = [];
  for (const o of outcomes) {
    if (!groupNames.includes(o.group)) {
      groupNames.push(o.group);
    }
  }
  for (const name of groupNames) {
    let groupTotal = 0;
    let groupTop = 0;
    let groupFirst = 0;
    for (const o of outcomes) {
      if (o.group === name) {
        groupTotal++;
        if (o.ok) {
          groupTop++;
        }
        if (o.rank === 1) {
          groupFirst++;
        }
      }
    }
    const lineColor = groupTop === groupTotal && groupFirst === groupTotal ? GREEN : groupTop === groupTotal ? YELLOW : RED;
    console.log(paint(lineColor, "  " + name + ": " + groupTop + " of " + groupTotal + " passed, " + groupFirst + " of " + groupTotal + " had the right chunk first"));
  }
  console.log("");

  // The questions where the right chunk was not first (a real finding for the team)
  let notFirstCount = 0;
  for (const o of outcomes) {
    if (o.rank !== 1) {
      notFirstCount++;
    }
  }
  if (notFirstCount > 0) {
    console.log(paint(CYAN, "QUESTIONS WHERE THE RIGHT CHUNK WAS NOT FIRST"));
    for (const o of outcomes) {
      if (o.rank !== 1) {
        const status = o.ok ? paint(YELLOW, "passed") : paint(RED, "FAILED");
        console.log("  - \"" + o.question + "\" (" + status + ")");
        console.log("      Should have been first: " + o.expectedName);
        console.log("      Was first instead:      " + o.firstPlaceName);
        if (o.rank === 0) {
          console.log("      The right chunk was not found at all.");
        } else {
          console.log("      The right chunk was result number " + o.rank + ".");
        }
      }
    }
    console.log("");
  }

  console.log("Note: " + total + " questions is a small sample. It shows how search behaves, not a guarantee.");
}

// Run the function
main();