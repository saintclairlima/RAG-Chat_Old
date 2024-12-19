from bert_score import score

# Example sentences
candidates = ["Legislatura é um período de quatro anos"]#, "children love cats"]
references = ["Os deputados se reúnem no plenário"]#, "kids are fond of felines"]

# Calculate BERTScore
P, R, F1 = score(candidates, references, lang="pt-br", verbose=True)

# Output results
print(f"Precision: {P.mean().item():.4f}")
print(f"Recall: {R.mean().item():.4f}")
print(f"F1 Score: {F1.mean().item():.4f}")