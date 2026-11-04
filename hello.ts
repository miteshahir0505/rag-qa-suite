// Load the secret key from the .env file
import "dotenv/config";

// Load the library that talks to the AI
import OpenAI from "openai";

// The AI model we use. To try another model, change only this line.
const MODEL = "openai/gpt-oss-20b";

// Set up the connection to Groq, using your key
const client = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});

async function sayHello() {
  // First check that the key was loaded (we print only yes or no, never the key itself)
  console.log("Key loaded from .env: " + (process.env.GROQ_API_KEY ? "yes" : "NO"));

  try {
    // Send one short question to the AI
    const response = await client.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: "Say hello in five words." }],
    });

    // Show which model answered, and what it said
    console.log("Model: " + response.model);
    console.log("AI says: " + response.choices[0].message.content);
  } catch (error: any) {
    // If the call fails, show a short version of what went wrong
    console.log("The AI call failed: " + String(error.message).slice(0, 200));
  }
}

// Run the function
sayHello();