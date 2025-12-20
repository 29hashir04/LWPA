# Firebase Configuration Template

This file provides templates and instructions for setting up Firebase integration with your Gait Recognition System.

## Firebase Project Setup

### 1. Create a Firebase Project
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project"
3. Enter project name (e.g., "gait-recognition-system")
4. Enable Google Analytics (optional)
5. Create the project

### 2. Enable Authentication
1. In Firebase Console, go to "Authentication" > "Sign-in method"
2. Enable "Email/Password" authentication
3. Optionally enable other providers (Google, Facebook, etc.)

### 3. Create Firestore Database
1. Go to "Firestore Database" > "Create database"
2. Choose "Start in test mode" (for development)
3. Select a location for your database
4. Create the database

### 4. Enable Storage (Optional)
1. Go to "Storage" > "Get started"
2. Choose "Start in test mode"
3. Select a location for your storage bucket

## Configuration Files

### Option 1: Streamlit Secrets (Recommended for Production)

Create a `.streamlit/secrets.toml` file in your project root:

```toml
[firebase]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYour-Private-Key-Here\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project-id.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project-id.iam.gserviceaccount.com"
api_key = "your-web-api-key"
messaging_sender_id = "your-messaging-sender-id"
app_id = "your-app-id"
```

### Option 2: Environment Variables

Set these environment variables:

```bash
export FIREBASE_SERVICE_ACCOUNT_PATH="/path/to/your/service-account-key.json"
export FIREBASE_PROJECT_ID="your-project-id"
export FIREBASE_API_KEY="your-web-api-key"
```

### Option 3: Service Account JSON File

1. Go to Firebase Console > Project Settings > Service Accounts
2. Click "Generate new private key"
3. Download the JSON file
4. Place it in your project directory as `firebase-service-account.json`
5. Update the path in `firebase_config.py`

## Firestore Security Rules

Create these security rules in Firestore Database > Rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own data
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Gait data is user-specific
    match /gait_data/{gaitId} {
      allow read, write: if request.auth != null && 
        request.auth.uid == resource.data.user_id;
      allow create: if request.auth != null && 
        request.auth.uid == request.resource.data.user_id;
    }
  }
}
```

## Storage Security Rules (if using Storage)

```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /gait_data/{userId}/{allPaths=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

## Testing the Integration

### 1. Test Authentication
- Run the app: `streamlit run optimized_gait_app_refactored.py`
- Try creating a new account
- Test login/logout functionality

### 2. Test Data Storage
- Enroll a gait pattern
- Check Firestore Console to see if data is saved
- Test data retrieval and deletion

### 3. Monitor Usage
- Check Firebase Console > Usage tab
- Monitor authentication and Firestore usage
- Set up billing alerts if needed

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Check if Email/Password is enabled in Firebase Console
   - Verify API keys are correct
   - Ensure service account has proper permissions

2. **Firestore Permission Denied**
   - Check security rules
   - Ensure user is authenticated
   - Verify user ID matches document user_id

3. **Import Errors**
   - Install required packages: `pip install firebase-admin pyrebase4`
   - Check Python version compatibility
   - Verify Firebase SDK versions

4. **Configuration Not Found**
   - Check secrets.toml file location
   - Verify environment variables
   - Ensure service account JSON is valid

### Debug Mode

Enable debug logging by adding this to your app:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Migration from Local Auth

If migrating from the local authentication system:

1. **Backup existing data**
   - Export local gait data
   - Save user_data.json file

2. **Migrate users**
   - Create Firebase accounts for existing users
   - Import gait data to Firestore

3. **Update imports**
   - Replace `simple_auth_system` imports with `firebase_auth_service`
   - Update authentication checks

## Production Considerations

1. **Security**
   - Use production Firestore rules
   - Enable App Check for additional security
   - Set up proper CORS policies

2. **Performance**
   - Use Firestore indexes for complex queries
   - Implement data pagination for large datasets
   - Consider caching strategies

3. **Monitoring**
   - Set up Firebase Performance Monitoring
   - Enable Crashlytics for error tracking
   - Monitor authentication metrics

4. **Backup**
   - Set up automated Firestore backups
   - Implement data export functionality
   - Create disaster recovery procedures
