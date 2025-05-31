import xgboost as xgb
import pandas as pd
import shap

# Charger le modèle XGBoost depuis un fichier JSON
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("structureModule/xgboost_model_esnnpperso.json")
expected_features = xgb_model.get_booster().feature_names


# === 2. Charger les features modifiées et les proba originales ===
df_mod = pd.read_csv("structure_attacks/experiment_1_message_id/selected_phishing_modified_domain_features.csv")  # ← CSV avec email_id + features
df_mod["num_received"] = 7.0
print(f"📊 Nombre de structures analysées : {len(df_mod)}")
#df_mod["num_ofuniquedomains_in_all_email_addresses"]=1.0
#df_mod["num_From_header_tags_in_thread"]=1.0
#df_mod["ratio_unique_To_domains_to_unique_domains_in_all_addresses"]=1.0
#df_mod["boundary_decimal"] = 0.0
#df_mod["ratio_unique_domains_to_domains_of_allURLs_in_anytags"]= 0.1
#df_mod["number_of_text_plain_sections"] = 1.0
# === 4. Prédire les nouvelles probabilités ===
X_mod = df_mod[expected_features]
proba = xgb_model.predict_proba(X_mod)
df_mod["phishing_probability_modified"] = proba[:, 1]

# === 5. Merge avec les probabilités originales ===
df_compare = df_mod

# === 6. Prédiction de label modifié et calculs ===
df_compare["predicted_label_modified"] = (df_compare["phishing_probability_modified"] > 0.5).astype(int)

# a. Pourcentage désormais légitimes
percent_legit_now = 100 * (df_compare["predicted_label_modified"] == 0).mean()

# b. Moyenne du drop dans les cas toujours détectés comme phishing
phishing_still = df_compare[df_compare["predicted_label_modified"] == 1].copy()
phishing_still["drop"] = phishing_still["phishing_probability_original"] - phishing_still["phishing_probability_modified"]
mean_drop = phishing_still["drop"].mean()

# === 7. Affichage des résultats ===
print(f"✅ {percent_legit_now:.2f}% des e-mails modifiés sont maintenant considérés comme légitimes.")
print(f"📉 Dans ceux encore détectés comme phishing, la probabilité moyenne a chuté de {mean_drop:.4f}.")

# === 8. Afficher les emails considérés comme légitimes maintenant ===
legit_emails = df_compare[df_compare["predicted_label_modified"] == 0][["email_id", "phishing_probability_modified"]]

print("\n📬 E-mails maintenant considérés comme légitimes :")
print(legit_emails.to_string(index=False))

# === 3. Créer l'explicateur SHAP ===
X_background = pd.read_csv("structureModule/final_train_dataset.csv").drop(columns=['label', 'index', 'email_id','source'], errors='ignore')

explainer = shap.Explainer(xgb_model, X_background)

# === 4. Calculer les valeurs SHAP ===
shap_values = explainer(X_mod)

# === 5. Graphiques SHAP ===

# a) Résumé global (features les plus importantes, toutes classes confondues)
shap.summary_plot(shap_values, X_mod)

# b) Barplot des impacts moyens par feature (classe phishing)
shap.plots.bar(shap_values.mean(0), max_display=30)

# === 4. Identifier les e-mails extrêmes ===
idx_highest = df_mod["phishing_probability_modified"].idxmax()
idx_lowest = df_mod["phishing_probability_modified"].idxmin()

print(f"\n📈 Email avec la plus HAUTE proba de phishing : index {idx_highest}")
print(f"Proba = {df_mod.loc[idx_highest, 'phishing_probability_modified']:.4f}")
print(f"\n📉 Email avec la plus BASSE proba de phishing : index {idx_lowest}")
print(f"Proba = {df_mod.loc[idx_lowest, 'phishing_probability_modified']:.4f}")

# 📈 Email avec la plus haute proba
print("\n🔥 Email avec la PLUS HAUTE probabilité de phishing :")
print(f"→ email_id: {df_mod.loc[idx_highest, 'email_id']} — proba: {df_mod.loc[idx_highest, 'phishing_probability_modified']:.4f}")
shap_values_high = explainer(X_mod.iloc[[idx_highest]])
df_shap_high = pd.DataFrame({
    "feature": X_mod.columns,
    "value": X_mod.iloc[idx_highest].values,
    "shap_value": shap_values_high.values[0]
}).sort_values(by="shap_value", key=abs, ascending=False)

print(df_shap_high.head(5))


# 📉 Email avec la plus basse proba
print("\n📦 Email avec la PLUS BASSE probabilité de phishing :")
print(f"→ email_id: {df_mod.loc[idx_lowest, 'email_id']} — proba: {df_mod.loc[idx_lowest, 'phishing_probability_modified']:.4f}")

shap_values_low = explainer(X_mod.iloc[[idx_lowest]])
df_shap_low = pd.DataFrame({
    "feature": X_mod.columns,
    "value": X_mod.iloc[idx_lowest].values,
    "shap_value": shap_values_low.values[0]
}).sort_values(by="shap_value", key=abs, ascending=False)

print(df_shap_low.head(5))


