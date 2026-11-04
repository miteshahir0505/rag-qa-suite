// Load the embedding library (it runs on your own PC, no key needed)
import { pipeline } from "@huggingface/transformers";

async function main() {
  // Tell the user the model is loading (the first run downloads about 25 MB)
  console.log("Loading the embedding model...");

  // Load the model that turns text into numbers
  const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  console.log("Model loaded.");

  // Two short sentences to turn into numbers
  const sentences = [
    "Welcome to the AI World.",
    "Now, you will learn how AI interact with the outside world",
    "We are checking the length of the sentences using RAG",
    "We just want to make sure that it works perfectly even if we change something",
  ];

  // Create the embeddings (mean pooling and normalize give one clean list per sentence)
  const output = await embedder(sentences, { pooling: "mean", normalize: true });

  // Turn the result into plain JavaScript lists
  const vectors = output.tolist();

  // Show how many lists we got, and how long each list is
  console.log("Number of sentences: " + vectors.length);
  console.log("Numbers per sentence: " + vectors[0].length);

  // Show the first 5 numbers of each sentence, rounded to 4 decimals
  for (let i = 0; i < vectors.length; i++) {
    const firstFive = vectors[i].slice(0, 5).map((n: number) => n.toFixed(4));
    console.log("Sentence " + (i + 1) + " starts with: " + firstFive.join(", "));
  }
}

// Run the function
main();