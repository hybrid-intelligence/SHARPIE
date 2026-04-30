# Server Setup Guide

This guide covers the one-time setup required on your production server before the CI/CD pipeline can deploy automatically.

## Prerequisites

- Ubuntu/Debian-based server (adjust commands for other distributions)
- Root or sudo access
- Domain name pointing to your server (for SSL)

## Step 1: Create Deployment User

Create a dedicated user for deployments (not root):

```bash
# Create user
sudo useradd -m -s /bin/bash sharpie-deploy

# Add to www-data group for web directory access
sudo usermod -a -G www-data sharpie-deploy

# Set password (optional, for manual login)
sudo passwd sharpie-deploy
```

## Step 2: Generate SSH Key for GitHub Actions

On your **local machine** (not the server):

```bash
# Generate dedicated SSH key for deployment
ssh-keygen -t rsa -b 4096 -f sharpie_deploy_key -C "github-actions-deploy"

# This creates two files:
# - sharpie_deploy_key (private key) -> Add to GitHub Secrets
# - sharpie_deploy_key.pub (public key) -> Add to server
```

Add the public key to the server:

```bash
# Option 1: Using ssh-copy-id (from your local machine)
ssh-copy-id -i sharpie_deploy_key.pub sharpie-deploy@your-server

# Option 2: Manually (if ssh-copy-id not available)
cat sharpie_deploy_key.pub | ssh sharpie-deploy@your-server 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && chmod 700 ~/.ssh'
```

Keep the private key for later (GitHub Secrets setup):
```bash
# Display private key (copy entire output including BEGIN/END lines)
cat sharpie_deploy_key
```

## Step 3: Install System Dependencies

On the server:

```bash
# Update package list
sudo apt-get update

# Install Python and dependencies
sudo apt-get install -y python3 python3-pip python3-venv

# Install Redis (required for Django Channels)
sudo apt-get install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Install Graphviz (required for pygraphviz)
sudo apt-get install -y graphviz libgraphviz-dev

# Install Supervisor (process manager)
sudo apt-get install -y supervisor
sudo systemctl enable supervisor
sudo systemctl start supervisor

# Install Nginx (web server/reverse proxy)
sudo apt-get install -y nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

## Step 4: Set Up Application Directory

```bash
# Create directory structure
sudo mkdir -p /var/www/sharpie

# Set ownership and permissions
sudo chown -R sharpie-deploy:www-data /var/www/sharpie
sudo chmod -R 775 /var/www/sharpie

# Switch to deployment user
sudo su - sharpie-deploy

# Clone repository
cd /var/www/sharpie
git clone https://github.com/YOUR_USERNAME/SHARPIE.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Exit from sharpie-deploy user
exit
```

## Step 5: Configure Supervisor Socket Permissions

Add the deployment user to the supervisor group to allow running `supervisorctl` without sudo:

```bash
# Create supervisor group if it doesn't exist
sudo groupadd supervisor 2>/dev/null || true

# Add deployment user to supervisor group
sudo usermod -a -G supervisor sharpie-deploy

# Configure supervisor socket permissions
sudo sed -i 's/^sockchmod=.*/sockchmod=0770/' /etc/supervisor/supervisord.conf
sudo sed -i 's/^sockchown=.*/sockchown=root:supervisor/' /etc/supervisor/supervisord.conf

# If the lines don't exist, add them under [unix_http_server] section
# Edit the file manually:
sudo nano /etc/supervisor/supervisord.conf
```

Add/ensure the following in `/etc/supervisor/supervisord.conf` under `[unix_http_server]`:

```ini
[unix_http_server]
file=/var/run/supervisor.sock
chmod=0770
chown=root:supervisor
```

Restart supervisor:

```bash
sudo systemctl restart supervisor
```

Verify the deployment user can access supervisor:

```bash
sudo su - sharpie-deploy
supervisorctl status
# Should show program status without permission error
exit
```

## Step 6: Configure Supervisor for SHARPIE

Copy and configure the supervisor configs:

```bash
# Copy configs to supervisor directory
sudo cp /var/www/sharpie/deployment/webserver_supervisor.conf /etc/supervisor/conf.d/sharpie-web.conf
sudo cp /var/www/sharpie/deployment/runner_supervisor.conf /etc/supervisor/conf.d/sharpie-runner.conf
```

Edit the configs to use correct paths:

```bash
# Edit webserver config
sudo nano /etc/supervisor/conf.d/sharpie-web.conf
```

Replace placeholder paths:
- Change `/my/app/path` to `/var/www/sharpie`
- Change `/path/to/venv` to `/var/www/sharpie/venv`
- Change `/your/log/` to `/var/www/sharpie/logs/`

Create log directory and apply configuration:

```bash
# Create log directory
sudo mkdir -p /var/www/sharpie/logs
sudo chown sharpie-deploy:www-data /var/www/sharpie/logs
sudo chmod 775 /var/www/sharpie/logs

# Create daphne socket directory
sudo mkdir -p /run/daphne
sudo chown sharpie-deploy:www-data /run/daphne

# Reload supervisor configuration
sudo supervisorctl reread
sudo supervisorctl update
```

## Step 7: Configure Nginx

```bash
# Copy nginx configuration
sudo cp /var/www/sharpie/deployment/nginx.conf /etc/nginx/sites-available/sharpie

# Edit the configuration
sudo nano /etc/nginx/sites-available/sharpie
```

Update the configuration:
- Replace `your-domain.com` with your actual domain
- Verify static files path matches your setup

Enable the site:

```bash
# Remove default site (optional)
sudo rm /etc/nginx/sites-enabled/default

# Enable SHARPIE site
sudo ln -s /etc/nginx/sites-available/sharpie /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

## Step 8: Set Up SSL Certificates (Let's Encrypt)

```bash
# Install Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Follow prompts to configure SSL
# Choose to redirect HTTP to HTTPS

# Test auto-renewal
sudo certbot renew --dry-run
```

## Step 9: Configure Database

For production, use PostgreSQL instead of SQLite:

```bash
# Install PostgreSQL
sudo apt-get install -y postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql

# In PostgreSQL shell:
CREATE DATABASE sharpie;
CREATE USER sharpie WITH PASSWORD 'your-secure-password';
ALTER ROLE sharpie SET client_encoding TO 'utf8';
ALTER ROLE sharpie SET default_transaction_isolation TO 'read committed';
ALTER ROLE sharpie SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE sharpie TO sharpie;
\q
```

Add `psycopg2-binary` to your requirements if using PostgreSQL:

```bash
sudo su - sharpie-deploy
cd /var/www/sharpie
source venv/bin/activate
pip install psycopg2-binary
```

## Step 10: Initial Database Setup

```bash
sudo su - sharpie-deploy
cd /var/www/sharpie
source venv/bin/activate
cd webserver

# Run migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic

# Create superuser (optional)
python manage.py createsuperuser

exit
```

## Step 11: Configure Environment Variables

Create environment file for Django settings:

```bash
# On the server as sharpie-deploy
sudo su - sharpie-deploy
cd /var/www/sharpie/webserver
nano .env
```

Add your production settings:

```env
DEBUG=False
SECRET_KEY=your-very-secure-secret-key-here
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
DATABASE_URL=postgres://sharpie:your-secure-password@localhost/sharpie
```

## Step 12: Verify Setup

```bash
# Check supervisor status
sudo supervisorctl status

# Should show:
# sharpie-web:running
# sharpie-runner:running

# Check if application is responding
curl http://localhost:8000

# Check nginx
sudo systemctl status nginx

# Check from browser
# Visit: https://your-domain.com
```

## Troubleshooting

### Permission Errors

```bash
# Fix ownership
sudo chown -R sharpie-deploy:www-data /var/www/sharpie

# Fix permissions
sudo chmod -R 775 /var/www/sharpie

# Fix supervisor socket
sudo chmod 770 /var/run/supervisor.sock
sudo chown root:supervisor /var/run/supervisor.sock
```

### Supervisor Issues

```bash
# Check supervisor logs
sudo tail -f /var/log/supervisor/supervisord.log

# Check program logs
sudo supervisorctl tail -f sharpie-web
sudo supervisorctl tail -f sharpie-runner

# Manually start programs
sudo supervisorctl start sharpie-web
sudo supervisorctl start sharpie-runner
```

### Nginx Issues

```bash
# Check error logs
sudo tail -f /var/log/nginx/error.log

# Test configuration
sudo nginx -t

# Reload
sudo systemctl reload nginx
```

### Redis Issues

```bash
# Check if Redis is running
sudo systemctl status redis-server

# Test connection
redis-cli ping
# Should return: PONG

# Check Redis logs
sudo journalctl -u redis-server
```

## Next Steps

After server setup is complete:
1. Configure GitHub Secrets (see [GITHUB_SECRETS.md](GITHUB_SECRETS.md))
2. Push to `main` branch to trigger first deployment
3. Monitor the deployment in GitHub Actions
4. Verify the application is working correctly