# GitHub Secrets Configuration

This guide explains how to configure GitHub Secrets for the CI/CD pipeline.

## Overview

GitHub Secrets store sensitive information that should not be committed to the repository. The deployment workflow uses these secrets to securely connect to your server.

## Required Secrets

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `DEPLOY_KEY` | Private SSH key for authentication | See below |
| `PRODUCTION_HOST` | Server IP address or domain | `123.45.67.89` or `sharpie.example.com` |
| `PRODUCTION_USER` | SSH username | `sharpie-deploy` |
| `PRODUCTION_URL` | Production URL for health checks | `https://sharpie.example.com` |

## Adding Secrets in GitHub

### Step 1: Navigate to Repository Settings

1. Go to your repository on GitHub
2. Click **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**

### Step 2: Add Each Secret

Add the following secrets one by one:

### DEPLOY_KEY

This is the private SSH key generated during server setup (see [SERVER_SETUP.md](SERVER_SETUP.md)).

```bash
# On your local machine, display the private key
cat sharpie_deploy_key
```

**When adding this secret:**

1. **Name**: `DEPLOY_KEY`
2. **Value**: Paste the entire contents of the private key file, including:
   ```
   -----BEGIN RSA PRIVATE KEY-----
   ...key contents...
   -----END RSA PRIVATE KEY-----
   ```

**Important:** Copy the entire output, including the BEGIN and END lines. Make sure there are no extra spaces or line breaks.

### PRODUCTION_HOST

1. **Name**: `PRODUCTION_HOST`
2. **Value**: Your server's IP address or domain name
   - Example IP: `123.45.67.89`
   - Example domain: `sharpie.example.com`

### PRODUCTION_USER

1. **Name**: `PRODUCTION_USER`
2. **Value**: The deployment username created on the server
   - Example: `sharpie-deploy`

### PRODUCTION_URL

1. **Name**: `PRODUCTION_URL`
2. **Value**: The full URL of your production application (for health checks)
   - Example: `https://sharpie.example.com`

**Note:** The URL should include the protocol (`https://`) and should have a health endpoint at `/health`, or modify the workflow to use your actual health check endpoint.

## Verifying Secrets

After adding secrets, verify they're correctly configured:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. You should see all four secrets listed (values are hidden)
3. The workflow will reference them as:
   - `${{ secrets.DEPLOY_KEY }}`
   - `${{ secrets.PRODUCTION_HOST }}`
   - `${{ secrets.PRODUCTION_USER }}`
   - `${{ secrets.PRODUCTION_URL }}`

## Security Best Practices

### SSH Key Security

- **Dedicated key**: Use a dedicated SSH key for deployment, not personal keys
- **Never commit**: Never commit the private key to the repository
- **Rotate regularly**: Consider rotating the deployment key periodically
- **Limit access**: Only the deployment user should have this key in authorized_keys

### Secret Access

- Secrets are only available to GitHub Actions workflows
- They are masked in workflow logs (appear as `***`)
- Forked repositories cannot access parent repository secrets
- Workflow runs from forks require approval to use secrets

### Principle of Least Privilege

- The deployment user should have minimal permissions
- Only necessary directories should be accessible
- Supervisor access via group, not root/sudo

## Updating Secrets

To update a secret:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click the **Update** button next to the secret
3. Enter the new value
4. Click **Update secret**

**Note:** Updates only affect future workflow runs, not currently running ones.

## Testing the Configuration

After setting up secrets and completing server setup, test the deployment:

### Option 1: Manual Trigger

1. Go to **Actions** tab in GitHub
2. Select **Deploy to Production** workflow
3. Click **Run workflow** → **Run workflow**
4. Monitor the workflow execution

### Option 2: Push to Main

1. Merge a pull request to the `main` branch
2. The workflow will automatically trigger
3. Monitor in the **Actions** tab

### Debugging Failed Deployments

If the deployment fails:

1. Go to **Actions** tab
2. Click on the failed workflow run
3. Expand each step to see detailed output
4. Common issues:
   - **Permission denied**: Check SSH key is correct
   - **Connection refused**: Verify server IP and user
   - **Health check failed**: Check if services started correctly on server

## Checklist

Before triggering the first deployment:

- [ ] Generated SSH key pair on local machine
- [ ] Added public key to server (`~/.ssh/authorized_keys`)
- [ ] Tested SSH connection manually
- [ ] Added `DEPLOY_KEY` secret (private key)
- [ ] Added `PRODUCTION_HOST` secret
- [ ] Added `PRODUCTION_USER` secret
- [ ] Added `PRODUCTION_URL` secret
- [ ] Verified all secrets are visible in GitHub settings
- [ ] Completed server setup (see [SERVER_SETUP.md](SERVER_SETUP.md))