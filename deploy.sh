#!/bin/bash

# Deploy TimeSeriesPro to Google Cloud Run
# Prerequisites: gcloud CLI installed and authenticated

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting deployment to Google Cloud Run...${NC}"

# Get current project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: No active project found. Set project with: gcloud config set project PROJECT_ID${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Project ID: $PROJECT_ID${NC}"

# Enable required APIs if not already enabled
echo -e "${YELLOW}🔧 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Submit build to Cloud Build
echo -e "${YELLOW}🏗️  Submitting build to Cloud Build...${NC}"
gcloud builds submit --config cloudbuild.yaml .

# Get the service URL
echo -e "${YELLOW}🔗 Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe timeseries-pro --region=us-central1 --format="value(status.url)" 2>/dev/null || echo "")

if [ -n "$SERVICE_URL" ]; then
    echo -e "${GREEN}✅ Deployment successful!${NC}"
    echo -e "${GREEN}🌐 Service URL: $SERVICE_URL${NC}"
    echo -e "${YELLOW}📊 View logs: gcloud run services logs tail timeseries-pro --region=us-central1${NC}"
else
    echo -e "${RED}❌ Deployment may have failed. Check the build logs above.${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 TimeSeriesPro is now running on Google Cloud Run!${NC}"