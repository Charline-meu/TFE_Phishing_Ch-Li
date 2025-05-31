# Phishing_Detection
1. Prend en input des json pas des csv donc : python input/csv_to_json.py -> output = 2 file json avec phishing et legitimate du csv de base

2. python train phishing.json phishing legitimate.json legitimate -> output sous forme de txt dans output/features

3. txt to csv : python txt_to_csv.py