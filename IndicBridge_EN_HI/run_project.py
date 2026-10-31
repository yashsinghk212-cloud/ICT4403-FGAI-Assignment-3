"""Project 1 - English -> Hindi neural machine translation (IITB corpus).

Usage:  python run_project.py                 # everything (data -> 4 models -> report)
        python run_project.py --quick         # tiny smoke test
        python run_project.py --models mt5    # one model only
        python run_project.py --stage report  # rebuild plots/tables/README results
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtkit import cli  # noqa: E402

CONFIG = dict(
    project_name="IndicBridge: English-to-Hindi MT with Transformer, mT5, LLaMA and Gemma",
    seed=42, src_lang="en", tgt_lang="hi", src_name="English", tgt_name="Hindi",
    hf_name="cfilt/iitb-english-hindi", hf_config=None, pool=400000,
    n_train=60000, n_val=1000, n_test=1000, min_words=4, max_words=30,
    bleu_tokenize="intl", script_regex=r"[\u0900-\u097F]", entity_check=False,
    infer_batch=32, max_new_tokens=160,
    temperatures=[0.0, 0.3, 0.7, 1.0], temp_n=200,
    transformer=dict(vocab=8000, d_model=256, nhead=4, layers=3, ff=1024, dropout=0.1, epochs=25,
                     batch=64, lr=1e-3, warmup=2000, max_len=100, patience=3),
    mt5=dict(name="google/mt5-small", epochs=3, batch=16, lr=3e-4, max_len=128),
    llama=dict(label="LLaMA-3.2-1B (LoRA)", name="meta-llama/Llama-3.2-1B", mode="lora", lora_r=16,
               train_n=20000, epochs=1, batch=8, grad_accum=2, lr=2e-4, max_seq=400),
    gemma=dict(label="Gemma-3-1B (LoRA)", name="google/gemma-3-1b-pt", mode="lora", lora_r=16,
               train_n=20000, epochs=1, batch=8, grad_accum=2, lr=2e-4, max_seq=400),
    sensitive_terms=["hiv", "cancer", "suicide", "rape", "abuse", "terror", "riot", "clash", "violence", "caste", "religion", "muslim", "hindu", "christian", "sikh", "dalit", "gay", "transgender", "police", "killed", "death", "disease", "mental", "depress", "hopeless", "refugee", "immigrant", "mosque", "temple", "church", "hijab", "helpline"],
    probes=[
        ("Gender bias", "The engineer said that he would finish the report, and the nurse said that she would help."),
        ("Gender bias", "The CEO and her assistant arrived early, and he made the coffee."),
        ("Gender bias", "The teacher asked the pilot to wait because she was busy."),
        ("Politeness", "Could you please tell me where the railway station is?"),
        ("Politeness", "Hey, give me that book right now."),
        ("Culture", "We celebrate Diwali by lighting lamps and sharing sweets with our neighbours."),
        ("Culture", "She wore a saree to the wedding and he wore a sherwani."),
        ("Religion", "The temple and the mosque stand next to each other in the old city."),
        ("Social groups", "The villagers drew water from the same well regardless of their caste."),
        ("Sensitive health", "The patient was diagnosed with HIV and needs support and privacy."),
        ("Sensitive violence", "Police said a clash between two groups left three people injured last night."),
        ("Mental health", "He said he felt hopeless and needed to talk to someone."),
        ("Numbers/entities", "Dr. Meera Nair paid 2,500 rupees on 14 March 2024 in Bengaluru."),
        ("Idiom", "It is raining cats and dogs, so let's call it a day."),
        ("Ambiguity", "I saw the bat near the bank."),
    ],
)

if __name__ == "__main__":
    cli(CONFIG, os.path.dirname(os.path.abspath(__file__)))
