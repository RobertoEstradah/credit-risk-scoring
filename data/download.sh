#!/usr/bin/env bash
# Requiere: pip install kaggle + token en ~/.kaggle/kaggle.json
# (aceptar reglas de la competencia en kaggle.com/c/home-credit-default-risk)
set -e
cd "$(dirname "$0")"
for f in application_train.csv bureau.csv previous_application.csv; do
  kaggle competitions download -c home-credit-default-risk -f "$f"
  unzip -o "$f.zip" && rm "$f.zip"
  echo "✓ $f listo"
done
