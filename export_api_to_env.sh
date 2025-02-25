#!/bin/bash

# Fixed values
FILE_PATH="gemini_api_key.txt"
ENV_VAR_NAME="GEMINI_API_KEY"

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File '$FILE_PATH' not found!"
    exit 1
fi

# Export file content to the environment variable
export "$ENV_VAR_NAME"="$(cat "$FILE_PATH")"

# Confirm the export
echo "Exported contents of '$FILE_PATH' to environment variable '$ENV_VAR_NAME'"