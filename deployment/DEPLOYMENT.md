# SHARPIE Deployment Guide

> **Note:** This document is intended for the development team maintaining the official SHARPIE deployment. If you are setting up a self-hosted instance, the steps may still be useful as a reference, but you will need to adapt them to your environment.

This document describes the CI/CD pipeline for automated deployment to production.

## Overview

SHARPIE uses GitHub Actions for continuous integration and deployment. When code is pushed to the `main` branch, the pipeline automatically:

1. Runs tests (on GitHub-hosted runner)
2. Deploys to the production server via self-hosted runner
3. Verifies the deployment with a health check

## Architecture

```
main push → Test Job (GitHub runner) → Deploy Job (Self-hosted runner) → Health Check
```

### Components

- **GitHub Actions**: Orchestrates the CI/CD pipeline
- **Self-Hosted Runner**: Runs on the production server for direct deployment
- **Supervisor**: Manages Django/Daphne and runner processes on the server
- **Nginx**: Reverse proxy for the webserver

## Files

| File | Purpose |
|------|---------|
| `.github/workflows/deploy-production.yml` | GitHub Actions workflow definition |
| `deployment/nginx.conf` | Nginx reverse proxy configuration |
| `deployment/webserver_supervisor.conf` | Supervisor config for Django/Daphne |
| `deployment/runner_supervisor.conf` | Supervisor config for the experiment runner |
| `deployment/SERVER_SETUP.md` | One-time server setup instructions |

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

2. **Deploy Job** (runs on self-hosted runner on production server)
   - Checkout code from main branch
   - Pull latest code from `main` branch
   - Update Python dependencies
   - Run database migrations
   - Collect static files
   - Restart services via supervisorctl

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

- Self-hosted runner runs as a dedicated user with limited permissions
- Deployment user has limited permissions on the server
- Supervisor access is granted via group membership, not sudo
- The runner only has access to repositories you explicitly grant

## Requirements

### System Dependencies (apt packages)

These must be installed on the server:

| Package | Purpose |
|---------|---------|
| `redis-server` | WebSocket channels backend for Django Channels |

### Python Dependencies

Installed automatically from `requirements.txt` during deployment.

### External Services

- **Redis**: Required for Django Channels message passing
- **PostgreSQL** (recommended): Production database (SQLite not recommended for production)
- **SSL certificates**: Managed via Certbot/Let's Encrypt

## Troubleshooting

### Common Issues

**Deployment fails: "Permission denied"**
- Check the deployment user has access to `/var/www/sharpie`
- Verify user is in the correct groups (`www-data`, `supervisor`)
- Check file permissions: `ls -la /var/www/sharpie`

**Services don't restart**
- Verify supervisor is running: `sudo systemctl status supervisor`
- Check supervisor logs: `sudo supervisorctl tail sharpie-web`
- Verify user is in the supervisor group

**Database migrations fail**
- Check database connection settings
- Verify database user permissions
- Review migration conflicts: `python manage.py showmigrations`

**Runner fails to start**
- Check runner status: `sudo ./svc.sh status` (in actions-runner directory)
- Review runner logs: `journalctl -u actions.runner.SHARPIE.sharpie-production -f`
- Verify runner is registered in GitHub Settings → Actions → Runners

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
- [Django Deployment Checklist](https://docs.djangoproject.com/en/stable/howto/deployment/checklist/)
- [Channels Deployment Guide](https://channels.readthedocs.io/en/stable/deploying.html)