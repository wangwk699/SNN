from datasets import load_dataset 
tldr_dataset = load_dataset("trl-lib/tldr")
print(tldr_dataset)

tulu_dataset = load_dataset("allenai/tulu-3-sft-mixture")
print(tulu_dataset)