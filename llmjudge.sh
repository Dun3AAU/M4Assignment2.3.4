#!/bin/bash
#SBATCH --job-name=llmjudge
#SBATCH --output=logs/llmjudge_output_%j.txt
#SBATCH --error=logs/llmjudge_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1


# 1. Loading environment
source .venv/bin/activate


# 2. Start vLLM in the background 
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 &
    
# Capture the Process ID so we can kill it later
VLLM_PID=$!

# 3. Wait for the server to be ready
echo "Waiting for vLLM to load the model..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    echo "Still waiting..."
    sleep 10
done
echo "vLLM server is up and ready!"

# 4. running script
echo "Running LLM evaluation script..."
python llmJudge.py

# 5. Clean up
echo "Evaluation complete. Shutting down vLLM server..."
kill $VLLM_PID