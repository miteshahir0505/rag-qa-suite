// Load the secret key from the .env file
import "dotenv/config";

// Load the library that talks to the AI
import OpenAI from "openai";

// Load a built-in helper that reads and writes files
import * as fs from "fs";

// The AI model that JUDGES the answers. It is a bigger model than the writer (gpt-oss-20b).
const JUDGE_MODEL = "openai/gpt-oss-120b";

// The answer files made by the earlier scripts (they are only read here, never changed)
const BASELINE_FILE = "writer-answers.json";
const NO_REFUSAL_FILE = "writer-answers-no-refusal.json";
const TOP1_FILE = "writer-answers-top1.json";

// The file where the judge's results are saved (each run replaces the old file)
const RESULTS_FILE = "judge-results.json";

// Milliseconds to rest between calls, so the free Groq tier is not flooded
const PAUSE_MS = 3000;

// How many times one call is tried again when Groq says "too many requests"
const MAX_TRIES = 6;

// The most words the judge may use up, including its hidden thinking
const MAX_OUTPUT = 3000;

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

// The rules we give the judge: what to decide, and the exact answer format
const judgeRules =
  "You are a strict fact checker. You are given numbered pieces of company documents, a question, and an answer that someone wrote from those pieces. " +
  "Decide whether the answer is faithful to the pieces. Judge ONLY against the pieces given. Ignore anything you know from elsewhere. " +
  "VERDICT rules: the answer is SUPPORTED only if every fact in it is stated in the pieces (different wording is fine). " +
  "It is NOT SUPPORTED if it contains a fact, number or conclusion that the pieces do not state, or that the pieces do not prove. " +
  "For example, saying something is 'not supported' just because the pieces do not mention it is NOT SUPPORTED. " +
  "A plain statement that the documents do not cover the question is SUPPORTED when the pieces really do not contain the answer. " +
  "EXTRA rules: if the answer says the documents do not cover the question and then adds more text, judge that added text. " +
  "RELEVANT means the added text helps with the question that was asked. " +
  "MISLEADING means the added text is not about what was asked, so a reader could take it as the answer. " +
  "If nothing is added after a refusal, or the answer is not a refusal, the extra is NONE. " +
  "Answer in exactly this format, with these three labels, each on its own line: " +
  "VERDICT: (SUPPORTED or NOT SUPPORTED) " +
  "EXTRA: (NONE or RELEVANT or MISLEADING) " +
  "REASON: (one plain sentence)";

// One thing to judge: where it came from, the question, the pieces, the answer.
// expectedVerdict is only set for the planted answers, where we already know the right verdict.
type Item = {
  set: string;
  question: string;
  pieces: { source: string; section: string; text: string }[];
  answer: string;
  status: string;
  expectedVerdict: string;
  plantedLabel: string;
};

// Waits for a short time
function wait(milliseconds: number) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
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

// Asks the judge. Returns the judge's text and a status:
//   ok = a real answer, empty = no answer, rate = Groq kept refusing, error = another problem
async function askJudge(prompt: string) {
  for (let attempt = 1; attempt <= MAX_TRIES; attempt++) {
    try {
      const response = await client.chat.completions.create({
        model: JUDGE_MODEL,
        messages: [
          { role: "system", content: judgeRules },
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

// Reads the verdict from the judge's text. Returns "SUPPORTED", "NOT SUPPORTED" or "" if unreadable.
function readVerdict(text: string) {
  const match = text.match(/VERDICT:\s*[*`]*\s*(NOT SUPPORTED|SUPPORTED)/i);
  return match ? match[1].toUpperCase() : "";
}

// Reads the extra rating from the judge's text. Returns "NONE", "RELEVANT", "MISLEADING" or "" if unreadable.
function readExtra(text: string) {
  const match = text.match(/EXTRA:\s*[*`]*\s*(NONE|RELEVANT|MISLEADING)/i);
  return match ? match[1].toUpperCase() : "";
}

// Reads the one-sentence reason from the judge's text
function readReason(text: string) {
  const match = text.match(/REASON:\s*(.*)/i);
  return match ? match[1].replace(/[*`]/g, "").trim() : "";
}

// Builds the message for the judge: the pieces, the question, the answer
function buildPrompt(item: Item) {
  let prompt = "Pieces of the documents:\n\n";
  for (let p = 0; p < item.pieces.length; p++) {
    prompt += "Piece " + (p + 1) + " (" + item.pieces[p].source + ", " + item.pieces[p].section + "): " + item.pieces[p].text + "\n\n";
  }
  prompt += "Question: " + item.question + "\n\n";
  prompt += "Answer to check: " + item.answer;
  return prompt;
}

// Reads a JSON file, or returns null if it is missing
function readJson(fileName: string) {
  if (!fs.existsSync(fileName)) {
    console.log(paint(RED, fileName + " was not found. Run the script that makes it first."));
    return null;
  }
  return JSON.parse(fs.readFileSync(fileName, "utf8"));
}

async function main() {
  console.log(paint(CYAN, "JUDGE: IS EACH ANSWER FAITHFUL TO THE DOCUMENTS IT WAS GIVEN?"));
  console.log("Question being tested: does every fact in the writer's answer appear in the pieces the writer received,");
  console.log("and when a refusal adds extra text, is that text relevant or misleading?");
  console.log("");
  console.log(paint(CYAN, "HOW TO READ THIS REPORT"));
  console.log("  - The judge (" + JUDGE_MODEL + ") is a different, bigger AI than the writer. It checks each answer against the pieces the writer was given.");
  console.log("  - The judge is an AI too, so it is tested FIRST on 4 planted answers whose correct verdict we already know.");
  console.log("    If it gets any of them wrong, the real answers are NOT judged, because the judge cannot be trusted.");
  console.log("  - PASS = every fact in the answer is in the pieces, and any added text after a refusal is relevant.");
  console.log("  - WARNING = every fact is in the pieces, but text added after a refusal is misleading (true, but not what was asked).");
  console.log("  - FAIL = the answer contains something the pieces do not support.");
  console.log("  - The judge checks against the pieces given, not against all the documents. A refusal can be faithful and still unhelpful if search gave the wrong pieces.");
  console.log("");

  // Load the three answer files
  const baseline = readJson(BASELINE_FILE);
  const noRefusal = readJson(NO_REFUSAL_FILE);
  const top1 = readJson(TOP1_FILE);
  if (baseline === null || noRefusal === null || top1 === null) {
    return;
  }

  // The planted answers use the pieces of two baseline questions (the first and the third)
  const first = baseline.answers[0];
  const third = baseline.answers[2];
  const planted: Item[] = [
    {
      set: "Planted check",
      plantedLabel: "good answer",
      question: first.question,
      pieces: first.pieces,
      answer: "Up to 10 units of one item.",
      status: "ok",
      expectedVerdict: "SUPPORTED",
    },
    {
      set: "Planted check",
      plantedLabel: "wrong number",
      question: first.question,
      pieces: first.pieces,
      answer: "Up to 20 units of one item.",
      status: "ok",
      expectedVerdict: "NOT SUPPORTED",
    },
    {
      set: "Planted check",
      plantedLabel: "correct fact plus an invented one",
      question: first.question,
      pieces: first.pieces,
      answer: "Up to 10 units of one item, and orders ship free.",
      status: "ok",
      expectedVerdict: "NOT SUPPORTED",
    },
    {
      set: "Planted check",
      plantedLabel: "good answer with a second correct fact",
      question: third.question,
      pieces: third.pieces,
      answer: "The reset link works for 1 hour and can be used only once.",
      status: "ok",
      expectedVerdict: "SUPPORTED",
    },
  ];

  // PART 1: test the judge on the planted answers
  console.log(paint(CYAN, "PART 1: TESTING THE JUDGE ON PLANTED ANSWERS"));
  console.log("");
  let plantedCorrect = 0;
  const savedPlanted: any[] = [];
  for (let i = 0; i < planted.length; i++) {
    const item = planted[i];
    const result = await askJudge(buildPrompt(item));
    const verdict = readVerdict(result.text);
    const reason = readReason(result.text);
    const ok = verdict === item.expectedVerdict;
    if (ok) {
      plantedCorrect++;
    }

    console.log(paint(MAGENTA, "Planted check " + (i + 1) + " of " + planted.length + " [" + item.plantedLabel + "]: " + item.question));
    console.log("  Requirement: the judge must give the correct verdict on an answer whose correct verdict we know");
    console.log("  Expected:    " + item.expectedVerdict);
    console.log("  Answer checked: " + item.answer);
    if (result.status !== "ok" || verdict === "") {
      console.log("  Actual:      the judge gave no readable verdict (status " + result.status + "): " + result.text.replace(/\s+/g, " ").slice(0, 150));
    } else {
      console.log("  Actual:      the judge said " + verdict + ". Reason: " + reason);
    }
    console.log("  Verdict:     " + (ok ? paint(GREEN, "PASS") : paint(RED, "FAIL")));
    console.log("  Technical detail: judge labels VERDICT " + (verdict || "none") + ", EXTRA " + (readExtra(result.text) || "none") + ", status " + result.status);
    console.log("");

    savedPlanted.push({ label: item.plantedLabel, answer: item.answer, expected: item.expectedVerdict, judgeVerdict: verdict, reason: reason, correct: ok });
    await wait(PAUSE_MS);
  }

  // If the judge got any planted answer wrong, stop here
  console.log(paint(CYAN, "PART 1 SUMMARY"));
  if (plantedCorrect !== planted.length) {
    console.log(paint(RED, "  The judge got " + plantedCorrect + " of " + planted.length + " planted answers right."));
    console.log(paint(RED, "  The judge cannot be trusted yet, so the real answers were NOT judged. Fix the judge rules first."));
    fs.writeFileSync(RESULTS_FILE, JSON.stringify({ judgeModel: JUDGE_MODEL, planted: savedPlanted, trusted: false, results: [] }, null, 2));
    return;
  }
  console.log(paint(GREEN, "  The judge got " + plantedCorrect + " of " + planted.length + " planted answers right, so it can be used on the real answers."));
  console.log("");

  // PART 2: judge the real answers. We gather them from the three files.
  const items: Item[] = [];
  for (const a of baseline.answers) {
    items.push({ set: "Main run (refusal rule present)", plantedLabel: "", question: a.question, pieces: a.pieces, answer: a.answer, status: a.status, expectedVerdict: "" });
  }
  for (const a of noRefusal.answers) {
    items.push({ set: "Scenario A (refusal rule removed)", plantedLabel: "", question: a.question, pieces: a.pieces, answer: a.answer, status: a.status, expectedVerdict: "" });
  }
  items.push({ set: "Scenario B (first piece only)", plantedLabel: "", question: top1.question, pieces: top1.pieces, answer: top1.answer, status: top1.status, expectedVerdict: "" });

  console.log(paint(CYAN, "PART 2: JUDGING THE REAL ANSWERS (" + items.length + " answers)"));
  console.log("");

  // Counters per set, and the lists of warnings and failures
  const counts: { [set: string]: { pass: number; warning: number; fail: number; notJudged: number } } = {};
  const problems: string[] = [];
  const savedResults: any[] = [];

  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (!counts[item.set]) {
      counts[item.set] = { pass: 0, warning: 0, fail: 0, notJudged: 0 };
    }

    console.log(paint(MAGENTA, "Answer " + (i + 1) + " of " + items.length + " [" + item.set + "]: " + item.question));
    console.log("  Requirement: every fact in the answer must be in the pieces the writer was given, and any text added after a refusal must be relevant");
    console.log("  Expected:    a faithful answer with nothing invented and nothing misleading added");
    console.log("  Answer checked: " + item.answer.replace(/\s+/g, " "));

    // An empty or failed writer answer cannot be judged
    if (item.status !== "ok" || item.answer.trim() === "") {
      console.log("  Actual:      the writer left no answer, so there is nothing to judge.");
      console.log("  Verdict:     " + paint(YELLOW, "NOT JUDGED"));
      console.log("");
      counts[item.set].notJudged++;
      savedResults.push({ set: item.set, question: item.question, answer: item.answer, verdict: "NOT JUDGED" });
      continue;
    }

    const result = await askJudge(buildPrompt(item));
    const judgeVerdict = readVerdict(result.text);
    const extra = readExtra(result.text);
    const reason = readReason(result.text);

    let finalVerdict = "";
    let verdictText = "";
    if (result.status !== "ok" || judgeVerdict === "" || extra === "") {
      console.log("  Actual:      the judge gave no readable answer (status " + result.status + "): " + result.text.replace(/\s+/g, " ").slice(0, 150));
      finalVerdict = "NOT JUDGED";
      verdictText = paint(YELLOW, "NOT JUDGED");
      counts[item.set].notJudged++;
    } else {
      console.log("  Actual:      the judge said " + (judgeVerdict === "SUPPORTED" ? "every fact is in the pieces" : "something in the answer is NOT in the pieces") + ". Reason: " + reason);
      if (extra === "MISLEADING") {
        console.log(paint(YELLOW, "  Extra text after the refusal: misleading (not about what was asked)"));
      } else if (extra === "RELEVANT") {
        console.log("  Extra text after the refusal: relevant to the question");
      }
      if (judgeVerdict === "NOT SUPPORTED") {
        finalVerdict = "FAIL";
        verdictText = paint(RED, "FAIL");
        counts[item.set].fail++;
        problems.push("FAIL    [" + item.set + "] " + item.question);
      } else if (extra === "MISLEADING") {
        finalVerdict = "WARNING";
        verdictText = paint(YELLOW, "WARNING");
        counts[item.set].warning++;
        problems.push("WARNING [" + item.set + "] " + item.question);
      } else {
        finalVerdict = "PASS";
        verdictText = paint(GREEN, "PASS");
        counts[item.set].pass++;
      }
    }
    console.log("  Verdict:     " + verdictText);
    console.log("  Technical detail: judge labels VERDICT " + (judgeVerdict || "none") + ", EXTRA " + (extra || "none") + ", status " + result.status);
    console.log("");

    savedResults.push({ set: item.set, question: item.question, answer: item.answer, judgeVerdict: judgeVerdict, extra: extra, reason: reason, verdict: finalVerdict });
    await wait(PAUSE_MS);
  }

  // Save the results (each run replaces the old file)
  fs.writeFileSync(RESULTS_FILE, JSON.stringify({ judgeModel: JUDGE_MODEL, trusted: true, planted: savedPlanted, results: savedResults }, null, 2));

  // THE SUMMARY, in plain sentences
  console.log(paint(CYAN, "SUMMARY"));
  console.log("  Judge check: " + plantedCorrect + " of " + planted.length + " planted answers judged correctly.");
  for (const setName of Object.keys(counts)) {
    const c = counts[setName];
    const total = c.pass + c.warning + c.fail + c.notJudged;
    const lineColor = c.fail > 0 ? RED : c.warning > 0 ? YELLOW : GREEN;
    let line = "  " + setName + ": " + c.pass + " of " + total + " PASS, " + c.warning + " WARNING, " + c.fail + " FAIL";
    if (c.notJudged > 0) {
      line += ", " + c.notJudged + " not judged";
    }
    console.log(paint(lineColor, line));
  }
  console.log("");
  if (problems.length > 0) {
    console.log(paint(CYAN, "ANSWERS THAT DID NOT PASS"));
    for (const p of problems) {
      console.log("  - " + p);
    }
    console.log("");
  }
  console.log("The judge's full results are saved in " + RESULTS_FILE + ".");
  console.log("Note: the judge is an AI and can be wrong. The 4 planted answers show it works on clear cases, not on every case.");
  console.log("Note: read the 'Reason' line of every WARNING and FAIL yourself before trusting the verdict.");
}

// Run the function
main();