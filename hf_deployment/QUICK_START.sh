#!/bin/bash
# QUICK_START.sh - 3-step deployment to Hugging Face Hub

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║          SLM → Hugging Face Hub Deployment (3 steps)          ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Install huggingface-hub if not already installed
echo "STEP 1️⃣  Installing dependencies..."
pip install huggingface-hub -q
echo "✓ Dependencies installed"
echo ""

# Step 2: Authenticate
echo "STEP 2️⃣  Authenticate with Hugging Face"
echo ""
echo "This will open a browser or prompt for your token."
echo "Get your token at: https://huggingface.co/settings/tokens"
echo ""
read -p "Press Enter to continue..." 

huggingface-cli login
echo ""

# Step 3: Upload
echo "STEP 3️⃣  Upload to Hugging Face Hub"
echo ""
read -p "Enter your Hugging Face username (without special chars): " USERNAME

if [ -z "$USERNAME" ]; then
    echo "❌ Username cannot be empty"
    exit 1
fi

REPO_ID="$USERNAME/slm"

echo ""
echo "Uploading to: https://huggingface.co/$REPO_ID"
echo ""

cd "$(dirname "$0")"
python upload_to_hf.py --repo_id "$REPO_ID" --create

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      ✅ DEPLOYMENT COMPLETE!                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Your model is now available at:"
echo "  🔗 https://huggingface.co/$REPO_ID"
echo ""
echo "Next:"
echo "  • Visit the link above to view your model"
echo "  • Edit the README.md on the model page to customize"
echo "  • Share the link with others!"
echo ""
