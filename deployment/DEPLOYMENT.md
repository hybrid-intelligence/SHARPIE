# SHARPIE Deployment Guide

This document describes the CI/CD pipeline for automated deployment to production.

## Overview

SHARPIE uses GitHub Actions for continuous integration and deployment. When code is pushed to the `main` branch, the pipeline automatically:

1. Runs tests
2. Deploys to the production server via SSH
3. Verifies the deployment with a health check

## Architecture

```
main push → Run Tests → SSH Deploy → Health Check
```

### Components

- **GitHub Actions**: Orchestrates the CI/CD pipeline
- **SSH Deployment**: Uses `appleboy/ssh-action` for remote deployment
- **Supervisor**: Manages Django/Daphne and runner processes on the server
- **Nginx**: Reverse proxy for the webserver

## Files

| File | Purpose |
|------|---------|
| `.github/workflows/deploy-production.yml` | GitHub Actions workflow definition |
| `deployment/nginx.conf` | Nginx reverse proxy configuration |
| `deployment/webserver_supervisor.conf` | Supervisor config for Django/Daphne |
| `deployment/runner_supervisor.conf` | Supervisor config for the WebSocket runner |
| `deployment/SERVER_SETUP.md` | One-time server setup instructions |
| `deployment/GITHUB_SECRETS.md` | GitHub Secrets configuration guide |

## Deployment Workflow

### Trigger Conditions

The deployment workflow triggers on:
- Push to `main` branch
- Manual workflow dispatch via GitHub Actions UI

### Workflow Steps

1. **Test Job** (runs on GitHub-hosted runner)
   - Checkout code
   - Set up Python 3.11
   - Install dependencies from `requirements.txt`
   - Run Django tests (`accounts`, `experiment`)

2. **Deploy Job** (runs after tests pass)
   - SSH to production server
   - Pull latest code from `main` branch
   - Update Python dependencies
   - Run database migrations
   - Collect static files
   - Restart services via supervisorctl

3. **Health Check**
   - Wait 10 seconds for services to start
   - Verify the application is responding

## Rollback Procedure

If a deployment causes issues:

1. SSH to the production server:
   ```bash
   ssh sharpie-deploy@your-server
   ```

2. Navigate to the application directory:
   ```bash
   cd /var/www/sharpie
   ```

3. Find the previous commit:
   ```bash
   git log --oneline -10
   ```

4. Checkout the previous commit:
   ```bash
   git checkout <previous-commit-hash>
   ```

5. Re-run deployment steps manually:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   cd webserver
   python manage.py migrate
   python manage.py collectstatic --noinput
   supervisorctl restart sharpie-web
   supervisorctl restart sharpie-runner
   ```

6. Verify application is working

7. Return to the main branch and fix the issue:
   ```bash
   git checkout main
   ```

## Security Considerations

- SSH deployment uses a dedicated key, not personal credentials
- GitHub Secrets store sensitive values, not in repository code
- Deployment user has limited permissions on the server
- Supervisor access is granted via group membership, not sudo

## Requirements

### System Dependencies (apt packages)

These must be installed on the server:

| Package | Purpose |
|---------|---------|
| `redis-server` | WebSocket channels backend for Django Channels |
| `graphviz` | Data model diagram generation |
| `libgraphviz-dev` | Development files for pygraphviz |

### Python Dependencies

Installed automatically from `requirements.txt` during deployment.

### External Services

- **Redis**: Required for Django Channels message passing
- **PostgreSQL** (recommended): Production database (SQLite not recommended for production)
- **SSL certificates**: Managed via Certbot/Let's Encrypt

## Troubleshooting

### Common Issues

**Deployment fails: "Permission denied"**
- Verify the SSH key is correctly added to GitHub Secrets
- Check the deployment user's authorized_keys on the server

**Services don't restart**
- Verify supervisor is running: `sudo systemctl status supervisor`
- Check supervisor logs: `sudo supervisorctl tail sharpie-web`
- Verify user is in the supervisor group

**Health check fails**
- Check if services are running: `supervisorctl status`
- Review application logs in `/var/www/sharpie/logs/`
- Verify nginx configuration: `sudo nginx -t`

**Database migrations fail**
- Check database connection settings
- Verify database user permissions
- Review migration conflicts: `python manage.py showmigrations`

### Useful Commands

```bash
# Check service status
supervisorctl status

# View service logs
supervisorctl tail -f sharpie-web
supervisorctl tail -f sharpie-runner

# Restart specific service
supervisorctl restart sharpie-web

# Check nginx status
sudo systemctl status nginx

# Test nginx configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx

# Check Redis status
sudo systemctl status redis-server

# Test Redis connection
redis-cli ping
```

## Monitoring

Consider setting up:
- GitHub Actions notifications (email, Slack, etc.)
- Server monitoring (CPU, memory, disk)
- Application performance monitoring
- Uptime monitoring (UptimeRobot, Pingdom)

## Further Reading

- [SERVER_SETUP.md](SERVER_SETUP.md) - One-time server configuration
- [GITHUB_SECRETS.md](GITHUB_SECRETS.md) - GitHub Secrets setup
- [Django Deployment Checklist](https://docs.djangoproject.com/en/stable/howto/deployment/checklist/)
- [Channels Deployment Guide](https://channels.readthedocs.io/en/stable/deploying.html)