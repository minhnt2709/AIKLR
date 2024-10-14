from transformers import BartForConditionalGeneration
from model_interface.tokenization_bart import (
    AMRBartTokenizer,
)  # We use our own tokenizer to process AMRs

model = BartForConditionalGeneration.from_pretrained(
    "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2"
)
tokenizer = AMRBartTokenizer.from_pretrained(
    "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2"
)

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(
    inputs["input_ids"], num_beams=2, min_length=0, max_length=20
)
print(summary_ids)
