# Code and Results for Master Thesis

In this repository, you will find the complete codebase for training and evaluating the model and its individual modules. It also includes the public dataset used, along with corresponding result files containing raw experimental outputs and, in some cases, scripts used to conduct the experiments.

Please note that some resources—such as the fine-tuned BERT model, *Enron* dataset and private email datasets (i.e., *Personal Advertising Emails* and *Personal Emails*)—are not publicly available due to confidentiality constraints and storage limitations.

The URL classification module is inspired by the approach proposed by Sahingoz et al. [1]. In particular, the feature extraction code for URLs was heavily inspired by the open-source implementation provided in this repository: [https://github.com/ebubekirbbr/pdd](https://github.com/ebubekirbbr/pdd). The overall model architecture is adapted from the D-Fence system introduced by Lee et al. [2].

## 📁 Project Structure

```
.
├── Datasets/               # Input datasets used for training/evaluation (without personal datasets)
├── EvaluateModel/          # Scripts for evaluating whole model performance
├── grapheShap/             # SHAP-based graph visualizations for structure module and URL module
├── Model/                  # Core model components
│   ├── metaClassifier/     # Meta-classifier combining three modules
│   ├── structureModule/    # Code for structure module
│   ├── textModule/         # Code for text module
│   └── urlsModule/         # Code for URL module
├── Results/                # Output results from experiments and attacks
│   ├── structure_attacks/  # Results and code for adversarial attacks on structure module
│   ├── text_attacks/       # Results of adversarial attacks on text module
│   └── urls_attacks/       # Results and code for adversarial attacks on URL module
├── auprc_plot.py           # Script for plotting AUPRC curves for SOTA
├── .gitignore              # Files and directories to ignore in Git (such as BERT model)
└── README.md               # Project overview (this file)
```

## 🧠 TODO

- Add more recent datasets
- Delevopping more robust models through adversarial training
- URL module: create a model that benefits from directly analysing the content and structure of linked webpage
- Text module: generalizing/removing/substiting names to help redue of overfitting
- **Conduct attacks on whole model: to combine attacks across different email components**

## 👤 Author

Liza Denis and Charline Meurant

## 📚 References

[1] Ozgur Koray Sahingoz, Ebubekir Buber, Onder Demir, and Banu Diri. *Machine learning based phishing detection from URLs*. Expert Systems with Applications, 117:345–357, 2019.

[2] Jehyun Lee, Farren Tang, Pingxiao Ye, Fahim Abbasi, Phil Hay, and Dinil Mon Divakaran. *D-Fence: A flexible, efficient, and comprehensive phishing email detection system*. In *2021 IEEE European Symposium on Security and Privacy (EuroS&P)*, pages 578–597. IEEE, 2021.