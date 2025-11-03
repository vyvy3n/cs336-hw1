# Text Generation Deliverable

## Generated Text (300 tokens)

**Prompt:** "Once upon a time"

**Generation parameters:** temperature=0.8, top-p=0.9, seed=42

**Model:** 4-layer Transformer trained for 40K iterations (lr=1e-3, val_loss=1.3881)

---

Once upon a time, there was a time, but he had to be alone. He was the sky, so, as he was a little did not getting ready for a group of a group of the sea, the sunbm lots of a young, the ship and he was when she was feeling really good.
Tim was the ship and his mom and was really important. "I know that he saw a little boy was very well, John was a day, he had not far away. They were a beautiful! He was going to find was going to be built. He was getting really hot air. He was playing at night. His parents had to go and the ship and the ship that day after that was when they went on the ship, and the ship up in the ship, no one day, it was going to look at seahiz us all the ship was at sea and soon they arrived at seah was going to shipmd was hot and the ship, they arrived at ship. They were when they had been away.
<|endoftext|>
Once upon a boy named Tom. Tom and loved music box was very far away, a big, there was very fast. He was a big, a small, it was his name, so much, he had a big, there was a little boy, a big, it was inside, it was very excited, there was a boy, Tom, it was very happy, "It was very hot, there was very old, it

---

## Fluency Analysis

The output shows **limited fluency** with significant coherence issues. While the model produces grammatically plausible local structures (e.g., "Once upon a time", "there was a little boy"), it struggles with narrative coherence, excessive repetition, and character confusion.

## Two Key Factors Affecting Output Quality

### Factor 1: Insufficient Training
The model was trained for only 40,000 iterations with a validation loss of 1.3881, indicating it has not fully converged. Evidence includes:
- Repetitive patterns ("the ship" appears 11 times in 300 tokens)
- Malformed tokens ("sunbm", "seahiz", "shipmd") suggesting incomplete vocabulary learning
- **Impact:** More training iterations would likely produce more coherent text with fewer malformed tokens

### Factor 2: Severely Limited Model Capacity (Single-Head Attention)
The model uses only **1 attention head** (vs. standard 8-16 heads in typical transformers). Evidence includes:
- Poor long-range coherence and inability to maintain consistent narrative threads
- Abrupt topic shifts (sky → sea → ship → music box) without logical transitions
- **Impact:** A single attention head cannot effectively model the diverse relationships needed for coherent story generation (character tracking, plot development, temporal relationships)

## Reproduction Command

```bash
python generate_text.py \
  --checkpoint checkpoints/lr_sweep/lr_1e_03/checkpoint_iter_40000.pt \
  --vocab artifacts/tinystories_vocab.json \
  --merges artifacts/tinystories_merges.txt \
  --prompt "Once upon a time" \
  --max-tokens 300 \
  --temperature 0.8 \
  --top-p 0.9 \
  --device cpu \
  --no-eos-stop \
  --seed 42
```

