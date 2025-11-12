# ==============================
# 🚀 Makefile: Deploy Python to GCP VM
# ==============================

# ---- CONFIG ----
PROJECT_PATH := $(CURDIR)
SCRIPT_NAME  := main.py                     
INSTANCE     := instance-20251112-001306   
ZONE         := asia-east1-c             
USER         := va2565_columbia_edu        
REMOTE_PATH  := ~/app                       
BINARY_NAME  := $(basename $(SCRIPT_NAME))  

# ---- TARGETS ----

.PHONY: all build upload run clean

all: build upload run

# Step 1: Build standalone binary
build:
	@echo "Building standalone binary using PyInstaller..."
	pip install pyinstaller --quiet
	pyinstaller --onefile $(SCRIPT_NAME)
	@echo "Binary created at dist/$(BINARY_NAME)"

# Step 2: Upload binary to VM
upload:
	@echo "Uploading binary to VM..."
	gcloud compute scp dist/$(BINARY_NAME) $(USER)@$(INSTANCE):$(REMOTE_PATH) --zone=$(ZONE)

# Step 3: SSH into VM and run
run:
	@echo "Running binary on VM..."
	gcloud compute ssh $(INSTANCE) --zone=$(ZONE) --command="chmod +x $(REMOTE_PATH)/$(BINARY_NAME) && $(REMOTE_PATH)/$(BINARY_NAME)"

# Step 4: Clean up local build files
clean:
	@echo "🧹 Cleaning build files..."
	rm -rf build dist *.spec
