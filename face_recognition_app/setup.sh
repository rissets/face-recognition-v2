#!/bin/bash

# Face Recognition Django App Setup Script
# This script sets up the complete environment for the face recognition application

set -e  # Exit on any error

echo "🚀 Starting Face Recognition Django App Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if virtual environment is activated
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_error "Virtual environment not activated!"
        print_status "Please activate your virtual environment first:"
        print_status "  source env/bin/activate"
        exit 1
    else
        print_status "Virtual environment activated: $VIRTUAL_ENV"
    fi
}

# Install Python dependencies
install_dependencies() {
    print_header "Installing Python Dependencies"
    
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        print_status "Dependencies installed successfully"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
}

# Setup environment variables
setup_environment() {
    print_header "Setting up Environment Variables"
    
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            print_status "Created .env file from .env.example"
            print_warning "Please edit .env file with your configurations!"
        else
            print_error ".env.example not found!"
            exit 1
        fi
    else
        print_status ".env file already exists"
    fi
}

# Generate Django secret key
generate_secret_key() {
    print_header "Generating Django Secret Key"
    
    # Generate a secure random secret key
    SECRET_KEY=$(python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())")
    
    # Update .env file
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/SECRET_KEY=your-secret-key-here-change-in-production/SECRET_KEY=$SECRET_KEY/" .env
    else
        # Linux
        sed -i "s/SECRET_KEY=your-secret-key-here-change-in-production/SECRET_KEY=$SECRET_KEY/" .env
    fi
    
    print_status "Django secret key generated and saved to .env"
}

# Generate encryption keys
generate_encryption_keys() {
    print_header "Generating Encryption Keys"
    
    # Generate 32-character encryption keys
    FACE_KEY=$(python -c "import secrets; print(secrets.token_hex(16))")
    PERSONAL_KEY=$(python -c "import secrets; print(secrets.token_hex(16))")
    JWT_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
    
    # Update .env file
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/FACE_DATA_ENCRYPTION_KEY=your-32-char-encryption-key-here/FACE_DATA_ENCRYPTION_KEY=$FACE_KEY/" .env
        sed -i '' "s/PERSONAL_DATA_ENCRYPTION_KEY=another-32-char-key-for-personal/PERSONAL_DATA_ENCRYPTION_KEY=$PERSONAL_KEY/" .env
        sed -i '' "s/JWT_SECRET_KEY=your-jwt-secret-key-here/JWT_SECRET_KEY=$JWT_KEY/" .env
    else
        # Linux
        sed -i "s/FACE_DATA_ENCRYPTION_KEY=your-32-char-encryption-key-here/FACE_DATA_ENCRYPTION_KEY=$FACE_KEY/" .env
        sed -i "s/PERSONAL_DATA_ENCRYPTION_KEY=another-32-char-key-for-personal/PERSONAL_DATA_ENCRYPTION_KEY=$PERSONAL_KEY/" .env
        sed -i "s/JWT_SECRET_KEY=your-jwt-secret-key-here/JWT_SECRET_KEY=$JWT_KEY/" .env
    fi
    
    print_status "Encryption keys generated and saved to .env"
}

# Check system dependencies
check_system_dependencies() {
    print_header "Checking System Dependencies"
    
    # Check for required system packages
    if ! command -v redis-server &> /dev/null; then
        print_warning "Redis not found. Installing..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS with Homebrew
            if command -v brew &> /dev/null; then
                brew install redis
            else
                print_error "Homebrew not found. Please install Redis manually."
                print_status "Visit: https://redis.io/download"
            fi
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y redis-server
            elif command -v yum &> /dev/null; then
                sudo yum install -y redis
            else
                print_error "Package manager not supported. Please install Redis manually."
            fi
        fi
    else
        print_status "Redis found"
    fi
    
    # Check for PostgreSQL (optional)
    if ! command -v psql &> /dev/null; then
        print_warning "PostgreSQL not found. Using SQLite for development."
        print_status "To use PostgreSQL, install it and update DATABASE_URL in .env"
    else
        print_status "PostgreSQL found"
    fi
}

# Setup database
setup_database() {
    print_header "Setting up Database"
    
    # Make migrations
    print_status "Creating migrations..."
    python manage.py makemigrations users
    python manage.py makemigrations core
    python manage.py makemigrations recognition
    python manage.py makemigrations analytics
    python manage.py makemigrations streaming
    
    # Apply migrations
    print_status "Applying migrations..."
    python manage.py migrate
    
    print_status "Database setup completed"
}

# Create superuser
create_superuser() {
    print_header "Creating Superuser"
    
    print_status "Creating Django superuser..."
    print_warning "You will be prompted to enter superuser details:"
    
    python manage.py createsuperuser
}

# Setup static files
setup_static_files() {
    print_header "Setting up Static Files"
    
    python manage.py collectstatic --noinput
    print_status "Static files collected"
}

# Create necessary directories
create_directories() {
    print_header "Creating Necessary Directories"
    
    mkdir -p media/profile_pictures
    mkdir -p media/face_images
    mkdir -p logs
    mkdir -p models/insightface
    
    print_status "Directories created"
}

# Download InsightFace models (optional)
download_models() {
    print_header "Setting up InsightFace Models"
    
    read -p "Do you want to download InsightFace models? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Downloading InsightFace models..."
        
        cd models/insightface
        
        # Download buffalo_l model (example)
        if [[ ! -f "buffalo_l.zip" ]]; then
            print_status "Downloading buffalo_l model..."
            wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
            unzip -q buffalo_l.zip
            rm buffalo_l.zip
            print_status "Buffalo_l model downloaded"
        else
            print_status "Buffalo_l model already exists"
        fi
        
        cd ../..
    else
        print_warning "Skipping model download. You can download them later."
        print_status "Visit: https://github.com/deepinsight/insightface/tree/master/model_zoo"
    fi
}

# Start services
start_services() {
    print_header "Starting Services"
    
    # Start Redis
    print_status "Starting Redis..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew services start redis
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo systemctl start redis-server
    fi
    
    print_status "Redis started"
    
    # Start ChromaDB (optional)
    read -p "Do you want to start ChromaDB server? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting ChromaDB..."
        nohup chroma run --host localhost --port 8000 > logs/chromadb.log 2>&1 &
        print_status "ChromaDB started on localhost:8000"
    fi
}

# Run tests
run_tests() {
    print_header "Running Tests"
    
    read -p "Do you want to run the test suite? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Running tests..."
        python manage.py test
        print_status "Tests completed"
    else
        print_warning "Skipping tests"
    fi
}

# Print final instructions
print_final_instructions() {
    print_header "Setup Complete! 🎉"
    
    echo
    print_status "Your Face Recognition Django App is ready!"
    echo
    print_status "Next steps:"
    echo "  1. Review and update .env file with your specific configurations"
    echo "  2. Start the development server:"
    echo "     python manage.py runserver"
    echo "  3. Start Celery worker (in another terminal):"
    echo "     celery -A face_app worker --loglevel=info"
    echo "  4. Start Celery beat (in another terminal):"
    echo "     celery -A face_app beat --loglevel=info"
    echo "  5. Access the admin interface at: http://localhost:8000/admin/"
    echo "  6. Access the API at: http://localhost:8000/api/"
    echo
    print_status "For more information, see README.md"
    echo
}

# Main setup function
main() {
    echo "🎯 Face Recognition Django App Setup"
    echo "===================================="
    echo
    
    # Run all setup steps
    check_venv
    install_dependencies
    setup_environment
    generate_secret_key
    generate_encryption_keys
    check_system_dependencies
    create_directories
    setup_database
    setup_static_files
    
    # Optional steps
    create_superuser
    download_models
    start_services
    run_tests
    
    # Final instructions
    print_final_instructions
}

# Run main function
main "$@"