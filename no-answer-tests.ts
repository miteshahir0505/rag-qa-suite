// Load a built-in helper that reads files
import * as fs from "fs";

// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

// The saved store we are testing
const STORE_FILE = "vector-store.json";

// The cut-off: a best score BELOW this means "the documents do not answer this question".
// This is a starting guess for this check, not an official value.
const CUTOFF = 0.30;

// A best score at or above this counts as a strong match (between the cut-off and this is a weak match).
// This is also a starting guess.
const STRONG = 0.45;

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

// Turns a score into a plain word, so nobody has to read numbers
function matchWord(score: number) {
  if (score >= STRONG) {
    return "strong match";
  }
  if (score >= CUTOFF) {
    return "weak match";
  }
  return "no real match";
}

// One test case.
// answerable false = the documents do NOT answer this question (the system should say "I don't know")
// answerable true  = the documents DO answer it (the system should give an answer)
// expectedSource and expectedSection (optional) = the exact part of the documents the answer must come from.
//   If they are given, using any other chunk counts as a FAIL.
type Test = {
  group: string;
  question: string;
  answerable: boolean;
  expectedSource?: string;
  expectedSection?: string;
};

const tests: Test[] = [
  // GROUP 1: questions about topics that are not in the documents at all
  {
    group: "Not covered: other topic",
    question: "What is your delivery time to another country?",
    answerable: false,
  },
  {
    group: "Not covered: other topic",
    question: "Do you offer gift wrapping?",
    answerable: false,
  },
  {
    group: "Not covered: other topic",
    question: "What are the opening hours of the support team?",
    answerable: false,
  },
  {
    group: "Not covered: other topic",
    question: "Which shipping address information do I need to give?",
    answerable: false,
  },

  // GROUP 2: questions with nothing to do with the product
  {
    group: "Not related at all",
    question: "What is the capital of France?",
    answerable: false,
  },
  {
    group: "Not related at all",
    question: "How do I bake a chocolate cake?",
    answerable: false,
  },

  // GROUP 3: questions that sound close to a document but ask for a detail the document does not give
  {
    group: "Sounds close, detail missing",
    question: "Can I log in with my fingerprint?",
    answerable: false,
  },
  {
    group: "Sounds close, detail missing",
    question: "Does the API support PUT requests to change an order?",
    answerable: false,
  },
  {
    group: "Sounds close, detail missing",
    question: "If the internet goes off while I pay, will the order still be processed?",
    answerable: false,
  },

  // GROUP 4: the documents only list what IS allowed, so the answer is implied.
  // A good system answers using the chunk that lists what is allowed.
  {
    group: "Answer is only implied",
    question: "Can I make payment using a gift card?",
    answerable: true,
    expectedSource: "checkout-prd.md",
    expectedSection: "Payment methods",
  },
  {
    group: "Answer is only implied",
    question: "Can I reset my password using my phone number?",
    answerable: true,
    expectedSource: "login-prd.md",
    expectedSection: "Password reset",
  },
  {
    group: "Answer is only implied",
    question: "If the payment is not made, will the order be completed?",
    answerable: true,
    expectedSource: "checkout-prd.md",
    expectedSection: "Failed payments",
  },

  // GROUP 5: a pair of questions that look alike. The wishlist is not in the documents, the cart is.
  {
    group: "Look-alike pair",
    question: "Does the wishlist hold more than 50 items?",
    answerable: false,
  },
  {
    group: "Look-alike pair",
    question: "Does the cart hold more than 50 items?",
    answerable: true,
    expectedSource: "add-to-cart-prd.md",
    expectedSection: "Quantity limits",
  },

  // GROUP 6: control group, questions the documents DO answer
  {
    group: "Control: documents do answer",
    question: "How many of the same product can I put in my basket?",
    answerable: true,
  },
  {
    group: "Control: documents do answer",
    question: "How long before an inactive user is logged out?",
    answerable: true,
  },
  {
    group: "Control: documents do answer",
    question: "How do I get my money back for a returned order?",
    answerable: true,
  },
  {
    group: "Control: documents do answer",
    question: "sold out",
    answerable: true,
  },
];

// One finished test, kept so the summary can group and explain the results
type Outcome = {
  group: string;
  question: string;
  answerable: boolean;
  bestScore: number;
  bestName: string;
  rightChunk: boolean;
  ok: boolean;
};

async function main() {
  // Explain the check in plain words for anyone reading the output
  console.log(paint(CYAN, "NO-ANSWER CHECK"));
  console.log("Question being tested: when the documents do NOT answer a question,");
  console.log("does the system notice, instead of showing an unrelated piece of text as if it were the answer?");
  console.log("");
  console.log(paint(CYAN, "HOW TO READ THIS REPORT"));
  console.log("  - The search always finds a \"best match\" in the documents, even when nothing in them answers the question.");
  console.log("  - Each best match is described in words: strong match, weak match, or no real match.");
  console.log("  - The system only answers when the best match is at least a weak match. Otherwise it replies:");
  console.log("    \"Sorry, I could not find this in the documents.\"");
  console.log("  - Questions the documents cannot answer: PASS = the system admits it does not know.");
  console.log("  - Questions the documents can answer: PASS = the system gives an answer.");
  console.log("  - Some questions are answered only by implication (the documents list what IS allowed, so what is not listed is not allowed).");
  console.log("    For these, PASS = the system answers using the part of the documents that lists what is allowed.");
  console.log("  - FAIL on a no-answer question = the system would show an unrelated piece of text as the answer.");
  console.log("  - FAIL on an answerable question = the system refuses, or answers from the wrong part of the documents.");
  console.log("  - The score behind each word is shown only on the small \"Technical detail\" line.");
  console.log("");

  // Load the saved store
  const store = JSON.parse(fs.readFileSync(STORE_FILE, "utf8"));
  const entries = store.entries;
  console.log("Store loaded: " + entries.length + " chunks (pieces of the documents).");

  // Load the same model that built the store
  console.log("Loading the embedding model...");
  const embedder = await pipeline("feature-extraction", store.model);
  console.log(paint(GREEN, "Model loaded.") + "\n");

  // All finished tests go here
  const outcomes: Outcome[] = [];

  for (let i = 0; i < tests.length; i++) {
    const test = tests[i];

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

    // The best match decides what the system would do
    const bestScore = scored[0].score;
    const bestName = "\"" + scored[0].section + "\" from " + scored[0].source;
    const wouldAnswer = bestScore >= CUTOFF;
    const word = matchWord(bestScore);

    // Some tests name the exact chunk the answer must come from
    const needsChunk = test.expectedSection !== undefined && test.expectedSource !== undefined;
    let rightChunk = true;
    if (needsChunk) {
      rightChunk = scored[0].source === test.expectedSource && scored[0].section === test.expectedSection;
    }
    const expectedName = needsChunk ? "\"" + test.expectedSection + "\" from " + test.expectedSource : "";

    // The verdict depends on what the question should get
    let ok = false;
    if (test.answerable) {
      ok = wouldAnswer && rightChunk;
    } else {
      ok = !wouldAnswer;
    }
    outcomes.push({
      group: test.group,
      question: test.question,
      answerable: test.answerable,
      bestScore: bestScore,
      bestName: bestName,
      rightChunk: rightChunk,
      ok: ok,
    });

    // Plain-words description of what should and did happen
    let expected = "";
    if (!test.answerable) {
      expected = "The system admits it does not know, because the documents do not cover this";
    } else if (needsChunk) {
      expected = "The system answers using " + expectedName + ", which covers this";
    } else {
      expected = "The system answers, using the part of the documents that covers this";
    }
    const actual = wouldAnswer
      ? "the system would ANSWER, using " + bestName + " (" + word + ")"
      : "the system would reply \"Sorry, I could not find this in the documents.\" (" + word + ")";

    // Show the test like a line in a QA report
    console.log(paint(MAGENTA, "Test " + (i + 1) + " of " + tests.length + " [" + test.group + "]: " + test.question));
    console.log("  Requirement: " + (test.answerable ? "a question the documents answer must get an answer" : "a question the documents cannot answer must not get an answer"));
    console.log("  Expected:    " + expected);
    console.log("  Actual:      " + actual);
    if (!test.answerable && wouldAnswer) {
      console.log(paint(YELLOW, "  Problem:     that part of the documents is unrelated to the question, but the system would still present it as the answer."));
    }
    if (test.answerable && !wouldAnswer) {
      console.log(paint(YELLOW, "  Problem:     the documents do contain the answer, but the system would refuse to give it."));
    }
    if (test.answerable && wouldAnswer && !rightChunk) {
      console.log(paint(YELLOW, "  Problem:     the system would answer, but from a different part of the documents than the one that covers this."));
    }
    console.log("  Verdict:     " + (ok ? paint(GREEN, "PASS") : paint(RED, "FAIL")));
    console.log("  Technical detail: closest match score " + bestScore.toFixed(2) + ", cut-off " + CUTOFF.toFixed(2));
    console.log("");
  }

  // Count the totals
  const total = outcomes.length;
  let passed = 0;
  for (const o of outcomes) {
    if (o.ok) {
      passed++;
    }
  }

  // THE SUMMARY, written as plain sentences
  console.log(paint(CYAN, "SUMMARY"));
  const summaryColor = passed === total ? GREEN : RED;
  console.log(paint(summaryColor, "  " + passed + " of " + total + " questions got the right treatment."));
  if (passed < total) {
    console.log(paint(RED, "  " + (total - passed) + " question(s) got the wrong treatment. They are listed below."));
  }
  console.log("");

  // Results per group
  console.log(paint(CYAN, "RESULTS PER GROUP"));
  const groupNames: string[] = [];
  for (const o of outcomes) {
    if (!groupNames.includes(o.group)) {
      groupNames.push(o.group);
    }
  }
  for (const name of groupNames) {
    let groupTotal = 0;
    let groupPassed = 0;
    for (const o of outcomes) {
      if (o.group === name) {
        groupTotal++;
        if (o.ok) {
          groupPassed++;
        }
      }
    }
    const lineColor = groupPassed === groupTotal ? GREEN : RED;
    console.log(paint(lineColor, "  " + name + ": " + groupPassed + " of " + groupTotal + " handled correctly"));
  }
  console.log("");

  // The questions that got the wrong treatment, in plain words
  let wrongCount = 0;
  for (const o of outcomes) {
    if (!o.ok) {
      wrongCount++;
    }
  }
  if (wrongCount > 0) {
    console.log(paint(CYAN, "QUESTIONS THAT GOT THE WRONG TREATMENT"));
    for (const o of outcomes) {
      if (!o.ok) {
        console.log("  - \"" + o.question + "\"");
        if (!o.answerable) {
          console.log(paint(RED, "      The documents do not answer this, but the system would answer with " + o.bestName + "."));
        } else if (o.bestScore < CUTOFF) {
          console.log(paint(RED, "      The documents answer this, but the system would refuse."));
        } else {
          console.log(paint(RED, "      The documents answer this, but the system would answer from the wrong part: " + o.bestName + "."));
        }
      }
    }
    console.log("");
  }

  // The key finding: can ONE rule separate the two kinds of question?
  // Find the no-answer question that came closest to being accepted, and the real question that came closest to being refused
  let closestNoAnswer: Outcome | null = null;
  let weakestReal: Outcome | null = null;
  for (const o of outcomes) {
    if (!o.answerable && (closestNoAnswer === null || o.bestScore > closestNoAnswer.bestScore)) {
      closestNoAnswer = o;
    }
    if (o.answerable && (weakestReal === null || o.bestScore < weakestReal.bestScore)) {
      weakestReal = o;
    }
  }
  console.log(paint(CYAN, "CAN ONE RULE SEPARATE THE TWO KINDS OF QUESTION?"));
  if (closestNoAnswer !== null && weakestReal !== null) {
    console.log("  The unanswerable question that came closest to being accepted:");
    console.log("    \"" + closestNoAnswer.question + "\" (" + matchWord(closestNoAnswer.bestScore) + ")");
    console.log("  The answerable question that came closest to being refused:");
    console.log("    \"" + weakestReal.question + "\" (" + matchWord(weakestReal.bestScore) + ")");
    if (closestNoAnswer.bestScore < weakestReal.bestScore) {
      console.log(paint(GREEN, "  Yes: on these questions, every question the documents cannot answer was a weaker match than every question they can answer."));
      console.log("  So one rule can separate them here. With more questions, this may stop being true.");
    } else {
      console.log(paint(RED, "  No: the unanswerable question above looked like a better match than the answerable one."));
      console.log("  So no single rule gets every question right. Making the rule stricter would refuse real questions, and making it looser would let unrelated text through.");
    }
    console.log("  Technical detail: closest unanswerable score " + closestNoAnswer.bestScore.toFixed(2) + ", weakest answerable score " + weakestReal.bestScore.toFixed(2));
  }
  console.log("");

  console.log("Note: " + total + " questions is a small sample, and the cut-off is a starting guess. It shows how search behaves, not a guarantee.");
}

// Run the function
main();