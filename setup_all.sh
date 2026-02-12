#!/bin/bash
echo "📦 Installing Frontend (Node) dependencies..."
npm install

echo "🐍 Setting up Python Virtual Environment..."
python -m venv venv
source venv/Scripts/activate || source venv/bin/activate

echo "🛠️ Installing Backend (Python) dependencies..."
pip install -r requirements.txt

echo "✅ Setup Complete. Use 'npm run dev' to start."