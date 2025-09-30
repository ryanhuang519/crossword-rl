import modal
import time

# Create Modal app
app = modal.App("gpt-oss-20b-inference")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # Install PyTorch nightly with CUDA 12.8
    .run_commands(
        "pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128"
    )
    # Install core stack
    .uv_pip_install(
        "torch>=2.8.0",
        "triton>=3.4.0",
        "numpy",
        "torchvision",
        "bitsandbytes",
        "transformers>=4.55.3",
    )
    # Install unsloth
    .uv_pip_install(
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
        "unsloth[base] @ git+https://github.com/unslothai/unsloth",
        "git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels",
    )
    # Pin transformers/tokenizers and install trl
    .uv_pip_install("transformers==4.56.2", "tokenizers", "trl==0.22.2")
)


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    cpu=1.0,  # 1 core = 2 vCPU
    memory=16384,  # 16 GiB
    timeout=3600,  # 1 hour timeout
)
def run_inference():
    """Run GPT-OSS-20B inference with a crossword puzzle example."""
    from unsloth import FastLanguageModel
    import torch
    from transformers import TextStreamer

    print("Loading model...")
    max_seq_length = 127000  # Can increase for longer RL output
    lora_rank = 4  # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b",
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        offload_embedding=True,  # Reduces VRAM by 1GB
    )

    print("\nPreparing crossword puzzle prompt...")
    messages = [
        {
            "role": "system",
            "content": """You're a crossword expert.
I will provide you with a 5x5 mini crossword and you should solve the entire puzzle in one go.

## Response format:
Provide all your guesses in a single message using the format: "guess 1a=red"
You can provide multiple guesses separated by commas, like: "guess 1a=red, guess 2d=blue, guess 3a=green"
You can also delete guesses you believe to be incorrect using "delete 1a"

DO NOT try to call a tool, simply respond with the response format. This is a multi-turn conversation.

## Important:
Try to solve the entire puzzle at once. Analyze all the clues together and provide your complete solution.
Think about how the across and down clues intersect and use those intersections to validate your answers.
Provide ALL your answers in ONE message to solve the puzzle as efficiently as possible."""
        },
        {
            "role": "user",
            "content": """# Crossword Puzzle Serialization
## Grid (5x5)
Legend:
- `black` = black square
- Number = clue label for the cell
- `.` = empty white square without a label
- Letter = filled entry
Grid layout:
Row1: col1 black, col2 01, col3 02, col4 03, col5 black.
Row2: col1 04, col2 ., col3 ., col4 ., col5 05.
Row3: col1 06, col2 ., col3 ., col4 ., col5 .
Row4: col1 07, col2 ., col3 ., col4 ., col5 .
Row5: col1 black, col2 08, col3 ., col4 ., col5 black.

## Clues

### Across
1A: Key above Caps Lock (3 letters)
4A: Biased sports fan (5 letters)
6A: What puts the "i" in Silicon Valley? (5 letters)
7A: Triangular road sign (5 letters)
8A: Items in a music library, for short (3 letters)

### Down
1D: Conversation subject (5 letters)
2D: Pumped up (5 letters)
3D: "Silver ___" (Christmas classic) (5 letters)
4D: Farm fodder (3 letters)
5D: Like pants in the classic Nantucket style (3 letters)"""
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )

    print("\nGenerating response...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    res = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        temperature=1.0,
        max_new_tokens=100000,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    decoded = tokenizer.decode(res[0], skip_special_tokens=True)
    print("\n" + "="*80)
    print("FINAL OUTPUT:")
    print("="*80)
    print(decoded)
    print("="*80)
    print(f"\nInference took {t1-t0:.2f} seconds")

    return decoded


@app.local_entrypoint()
def main():
    """Local entrypoint to run the inference."""
    result = run_inference.remote()
    print("\nInference completed successfully!")