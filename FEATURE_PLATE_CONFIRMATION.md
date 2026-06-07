# Plate Scan Confirmation Feature

## Overview
Implemented a confirmation workflow for plate scan operations that allows users to save or discard uploaded plate scan files and associated database records.

## Changes

### Models & Schemas
- **PlateConfirmRequest**: Request schema for plate confirmation action
  - `uploader_id`: Identifier of the uploader
  - `action`: Either "save" or "discard"
  - `saved_photo_filename`: Optional basename of the saved photo file
  - `saved_result_photo_filename`: Optional basename of the plate detection result photo

- **PlateConfirmResponse**: Response schema providing confirmation status
  - `uploader_id`: Echo of the request uploader ID
  - `action`: Echo of the action performed
  - `success`: Boolean indicating success/failure
  - `message`: Human-readable status message

### API Endpoints
- **POST /plate/confirm**: New confirmation endpoint
  - **Action "save"**: Retains uploaded files without modification
  - **Action "discard"**: 
    - Deletes photo files from configured directories
    - Removes plate scan records from the database
    - Reports errors for missing files or DB failures

### Improvements
- Modified `/plate/scan` response to return only filenames instead of full paths
- Provides better separation of concerns between scan and confirmation operations
- Implements robust error handling for file deletion and database cleanup

## Usage Example

```bash
# After scanning a plate with /plate/scan
curl -X POST "http://localhost:8000/plate/confirm" \
  -H "Content-Type: application/json" \
  -d {
    "uploader_id": "user123",
    "action": "discard",
    "saved_photo_filename": "20260607_123456_photo.jpg",
    "saved_result_photo_filename": "20260607_123456_result.jpg"
  }
```

## Database Impact
- No schema changes required
- Existing `/delete` logic reused via `delete_plate_results()`
- Supports idempotent operations (missing files do not cause failure)

## Error Handling
- Invalid actions rejected with clear error message
- File deletion failures logged without aborting the operation
- Database deletion errors captured and reported
- All errors aggregated in response message field
