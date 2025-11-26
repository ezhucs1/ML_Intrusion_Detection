#!/bin/bash

# Script to push CSE543_Group1 and rag_demo to GitHub
# This script will:
# 1. Move git repository to parent directory
# 2. Set up proper .gitignore
# 3. Add both projects
# 4. Commit and push to GitHub

set -e  # Exit on error

echo "=========================================="
echo "GitHub Push Script for ML Intrusion Detection"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the parent directory
PARENT_DIR="/home/ezhucs1/detection_ML"
CSE543_DIR="$PARENT_DIR/CSE543_Group1"
RAG_DEMO_DIR="$PARENT_DIR/rag_demo"
GITHUB_REPO="https://github.com/ezhucs1/ML_Intrusion_Detection.git"

# Step 1: Navigate to parent directory
echo -e "${YELLOW}Step 1: Checking directories...${NC}"
cd "$PARENT_DIR"

if [ ! -d "$CSE543_DIR" ]; then
    echo -e "${RED}Error: CSE543_Group1 directory not found!${NC}"
    exit 1
fi

if [ ! -d "$RAG_DEMO_DIR" ]; then
    echo -e "${RED}Error: rag_demo directory not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Directories found${NC}"
echo ""

# Step 2: Check if git is initialized in CSE543_Group1
echo -e "${YELLOW}Step 2: Setting up git repository...${NC}"
if [ -d "$CSE543_DIR/.git" ]; then
    echo "Git repository found in CSE543_Group1"
    
    # Move .git to parent if not already there
    if [ ! -d "$PARENT_DIR/.git" ]; then
        echo "Moving .git to parent directory..."
        mv "$CSE543_DIR/.git" "$PARENT_DIR/.git"
        echo -e "${GREEN}✓ Git repository moved to parent directory${NC}"
    else
        echo -e "${GREEN}✓ Git repository already in parent directory${NC}"
    fi
elif [ -d "$PARENT_DIR/.git" ]; then
    echo -e "${GREEN}✓ Git repository already in parent directory${NC}"
else
    echo "Initializing new git repository..."
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
fi
echo ""

# Step 3: Set up remote
echo -e "${YELLOW}Step 3: Configuring remote repository...${NC}"
git remote remove origin 2>/dev/null || true
git remote add origin "$GITHUB_REPO"
echo -e "${GREEN}✓ Remote configured: $GITHUB_REPO${NC}"
echo ""

# Step 4: Create/update .gitignore
echo -e "${YELLOW}Step 4: Setting up .gitignore...${NC}"
cat > "$PARENT_DIR/.gitignore" << 'EOF'
# CSE543_Group1 ignores
CSE543_Group1/data_original/
CSE543_Group1/Testing_data/
CSE543_Group1/models/*.pkl
CSE543_Group1/models_size_*/
CSE543_Group1/models_multiclass_*/
CSE543_Group1/visualizations/
CSE543_Group1/__pycache__/
CSE543_Group1/src/__pycache__/

# rag_demo ignores
rag_demo/venv/
rag_demo/chroma_db/
rag_demo/__pycache__/
rag_demo/src/__pycache__/

# Python
__pycache__/
*.py[cod]
*.pyc
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
*:Zone.Identifier
*.Zone.Identifier

# Logs
*.log

# Temporary files
*.tmp
*.temp

# Large training outputs
training_output.log

# Visualizations (generated files)
*.png
*.jpg
*.jpeg
*.pdf
EOF
echo -e "${GREEN}✓ .gitignore created${NC}"
echo ""

# Step 5: Clean up files that shouldn't be committed
echo -e "${YELLOW}Step 5: Cleaning up ignored files...${NC}"
# Remove __pycache__ directories if they exist
find "$CSE543_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$RAG_DEMO_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

# Step 6: Add files to git
echo -e "${YELLOW}Step 6: Adding files to git...${NC}"
git add .
echo -e "${GREEN}✓ Files added${NC}"
echo ""

# Step 7: Show what will be committed
echo -e "${YELLOW}Step 7: Files to be committed:${NC}"
echo "----------------------------------------"
git status --short | head -30
if [ $(git status --short | wc -l) -gt 30 ]; then
    echo "... and more files"
fi
echo "----------------------------------------"
echo ""

# Step 8: Ask for confirmation
read -p "Do you want to commit and push these changes? (yes/no): " confirm
if [ "$confirm" != "yes" ] && [ "$confirm" != "y" ]; then
    echo "Aborted. Files are staged but not committed."
    echo "You can review with: git status"
    echo "Then commit manually with: git commit -m 'Your message'"
    exit 0
fi

# Step 9: Commit
echo -e "${YELLOW}Step 8: Committing changes...${NC}"
git commit -m "Add CSE543_Group1 ML models and rag_demo for presentation demo

- Binary classification model with 100k training samples
- Multiclass classification model
- Testing scripts with graph generation
- RAG demo for interactive querying
- All necessary source files and documentation"
echo -e "${GREEN}✓ Changes committed${NC}"
echo ""

# Step 10: Set branch to main
echo -e "${YELLOW}Step 9: Setting up branch...${NC}"
git branch -M main 2>/dev/null || true
echo -e "${GREEN}✓ Branch set to main${NC}"
echo ""

# Step 11: Push to GitHub
echo -e "${YELLOW}Step 10: Pushing to GitHub...${NC}"
echo "Using HTTPS authentication..."
git remote set-url origin "$GITHUB_REPO"

# Try to push
if git push -u origin main; then
    echo -e "${GREEN}✓ Successfully pushed to GitHub!${NC}"
    echo ""
    echo "=========================================="
    echo -e "${GREEN}SUCCESS!${NC}"
    echo "=========================================="
    echo "Repository: $GITHUB_REPO"
    echo "Branch: main"
    echo ""
    echo "You can now clone on your laptop with:"
    echo "  git clone $GITHUB_REPO"
    echo ""
else
    echo -e "${RED}Push failed.${NC}"
    echo ""
    echo "Possible reasons:"
    echo "1. Authentication required (use Personal Access Token)"
    echo "2. Network issues"
    echo ""
    echo "To push manually:"
    echo "  git push -u origin main"
    echo ""
    echo "If authentication fails, use HTTPS with token:"
    echo "  Username: ezhucs1"
    echo "  Password: [Your GitHub Personal Access Token]"
    echo ""
    exit 1
fi


