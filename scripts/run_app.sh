#!/bin/bash
# AI Review Analyst - Streamlit 앱 실행 스크립트

cd "$(dirname "$0")/.."

echo "================================================"
echo "🔍 AI Review Analyst 시작"
echo "================================================"

# 가상환경 활성화
source ./venv/bin/activate

# Streamlit 앱 실행
streamlit run src/ui/app.py --server.port 8501
