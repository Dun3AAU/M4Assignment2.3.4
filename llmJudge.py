import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Define the Structured Output Schema using Pydantic
class LLMAssessment(BaseModel):
    llm_green_suggested: int = Field(
        description="1 if the claim describes green technology (e.g., climate change mitigation, renewable energy, clean tech), 0 if not."
    )
    llm_confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence level in the prediction."
    )
    llm_rationale: str = Field(
        description="1-3 sentences explaining the reasoning. You MUST cite specific phrases from the claim text."
    )

# 2. Connect to the local vLLM server via LangChain's OpenAI wrapper
# Make sure your vLLM server is running on localhost:8000
print("Connecting to vLLM server...")
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct", # Must match the model name running in vLLM
    openai_api_key="EMPTY",                      # Local vLLM doesn't need a real key
    openai_api_base="http://localhost:8000/v1",
    temperature=0.0,                             # 0.0 is best for strict classification
    max_tokens=256
)

# Bind the Pydantic schema to the LLM
structured_llm = llm.with_structured_output(LLMAssessment)

# 3. Create the Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert patent analyst. Evaluate the following patent claim. "
               "Rely ONLY on the provided text. Do not hallucinate external metadata. "
               "Provide your final assessment strictly in the requested format."),
    ("user", "Claim Text:\n{text}")
])

chain = prompt | structured_llm

# 4. Load your high-risk dataset
input_file = "data/hitl_green_100.csv"
print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file)

# Prepare lists to hold the new data
suggestions = []
confidences = []
rationales = []

# 5. Process each row
print("Evaluating claims with LLM...")
for text in tqdm(df['text'], desc="LLM Processing"):
    try:
        # Invoke the chain
        result = chain.invoke({"text": text})
        
        suggestions.append(result.llm_green_suggested)
        confidences.append(result.llm_confidence)
        rationales.append(result.llm_rationale)
        
    except Exception as e:
        # Fallback in case a smaller local model hallucinates outside the JSON schema
        print(f"Error processing text: {e}")
        suggestions.append(None)
        confidences.append("low")
        rationales.append(f"Error generating response: {str(e)}")

# 6. Append the new columns to the dataframe
df['llm_green_suggested'] = suggestions
df['llm_confidence'] = confidences
df['llm_rationale'] = rationales

# Reorder columns slightly so the human labeling columns are at the very end for easy Excel use
cols = ['doc_id', 'text', 'p_green', 'u', 'llm_green_suggested', 'llm_confidence', 'llm_rationale', 'human_label_is_green', 'human_notes']
df = df[cols]

# 7. Export the final file for Excel review
output_file = "data/hitl_green_100_ready_for_review.csv"
df.to_csv(output_file, index=False)
print(f"\nDone! File saved to {output_file}. Ready for manual Excel review.")