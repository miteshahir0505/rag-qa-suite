// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

// The score at or above which we treat two sentences as "the same meaning".
// This is a starting guess for this check, not an official value.
const MATCH_SCORE = 0.5;

// Compares two lists of numbers and gives one score (0 = nothing in common, 1 = same meaning).
// The model already scales each list to the same size, so multiplying and adding is enough.
function similarity(a: number[], b: number[]) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total = total + a[i] * b[i];
  }
  return total;
}

// The test cases. Each one says what a user would expect.
// expect "match" means: these two sentences should be treated as the same meaning.
// expect "no match" means: these two sentences should NOT be treated as the same meaning.
const tests = [
  {
    name: "Same meaning, different words (website problem)",
    a: "Our website login is broken.",
    b: "Users cannot sign in to the site.",
    expect: "match",
    why: "A user describing the problem in their own words should find the matching document.",
  },
  {
    name: "Same meaning, different words (everyday)",
    a: "The cat sat on the sofa.",
    b: "A kitten was resting on the couch.",
    expect: "match",
    why: "Different words, same idea. Search should not depend on exact words.",
  },
  {
    name: "Unrelated topics (website and money)",
    a: "Our website login is broken.",
    b: "The stock market fell sharply today.",
    expect: "no match",
    why: "A search about a login problem should not return a stock market document.",
  },
  {
    name: "Unrelated topics (pets and money)",
    a: "The cat sat on the sofa.",
    b: "The stock market fell sharply today.",
    expect: "no match",
    why: "Documents about different subjects should stay apart.",
  },
  {
    name: "Tricky: opposite meanings",
    a: "The payment went through successfully.",
    b: "The payment failed.",
    expect: "no match",
    why: "Someone searching for failed payments must not be shown successful ones.",
  },
  {
    name: "Tricky: same word, different meaning",
    a: "The bank approved my loan.",
    b: "We sat on the bank of the river.",
    expect: "no match",
    why: "The word 'bank' is shared, but the topics are not.",
  },
];

async function main() {
  // Explain the check in plain words for anyone reading the output
  console.log("EMBEDDING QUALITY CHECK");
  console.log("Question being tested: does the search model treat sentences with the");
  console.log("same meaning as a match, and sentences with different meanings as not a match?");
  console.log("Each test below has an expected result. The check compares it with the actual result.\n");

  console.log("Loading the model (the first run downloads it, about 25 MB)...");
  const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  console.log("Model loaded.\n");

  // Count the results as we go
  let passed = 0;
  const failedNames: string[] = [];

  // Run every test case
  for (let i = 0; i < tests.length; i++) {
    const test = tests[i];

    // Turn both sentences into numbers, then compare them
    const output = await embedder([test.a, test.b], { pooling: "mean", normalize: true });
    const vectors: number[][] = output.tolist();
    const score = similarity(vectors[0], vectors[1]);

    // What the model actually decided
    const actual = score >= MATCH_SCORE ? "match" : "no match";

    // The verdict: did the actual result equal the expected result?
    const ok = actual === test.expect;
    if (ok) {
      passed++;
    } else {
      failedNames.push(test.name);
    }

    // Show the test like a line in a QA report
    console.log("Test " + (i + 1) + ": " + test.name);
    console.log("  Sentence A: " + test.a);
    console.log("  Sentence B: " + test.b);
    console.log("  Why it matters: " + test.why);
    console.log("  Expected: " + test.expect);
    console.log("  Actual:   " + actual + " (score " + score.toFixed(2) + ")");
    console.log("  Verdict:  " + (ok ? "PASS" : "FAIL"));
    console.log("");
  }

  // The summary
  console.log("SUMMARY");
  console.log(passed + " of " + tests.length + " tests passed.");
  if (failedNames.length > 0) {
    console.log("Failed tests:");
    for (const name of failedNames) {
      console.log("  - " + name);
    }
    console.log("A FAIL is a finding about the model, not a coding mistake. A tester would");
    console.log("record it, judge how serious it is for the product, and decide what to do.");
  }
  console.log("\nNote: the pass mark of " + MATCH_SCORE + " is a starting guess. A real project would");
  console.log("choose it by testing many more examples.");
}

// Run the function
main();