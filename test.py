from transformers import pipeline
nlp = pipeline("fill-mask", model="vinai/phobert-large")
a = nlp(f"ông {nlp.tokenizer.mask_token}  30 năm làm nhà tình thương tôi gặp ông lê văn on ( tám dn ) ngụ ở ấp tiên long", targets=['già'])
print(a[0]['score'])