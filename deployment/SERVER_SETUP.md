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

## Step 2: Install System Dependencies

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

# Install Supervisor (process manager)
sudo apt-get install -y supervisor
sudo systemctl enable supervisor
sudo systemctl start supervisor

# Install Nginx (web server/reverse proxy)
sudo apt-get install -y nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

## Step 3: Set Up Application Directory

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

## Step 4: Configure Supervisor Socket Permissions

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

## Step 5: Configure Supervisor for SHARPIE

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

```bash
# Edit runner config
sudo nano /etc/supervisor/conf.d/sharpie-runner.conf
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

## Step 6: Configure Nginx

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

## Step 7: Set Up SSL Certificates (Let's Encrypt)

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

## Step 8: Configure Database

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

## Step 9: Initial Database Setup

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

## Step 10: Configure Environment Variables

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
PROD=True
DEMO=True
HTTPS=True
SECRET_KEY=
# IMPORTANT: Generate a secure secret key before running the application
# Generate one with: python -c "from secrets import token_urlsafe; print(token_urlsafe(50))"
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
REGISTRATION_KEY=your-registration-key
DATABASE_URL=postgres://sharpie:your-secure-password@localhost/sharpie
```

## Step 11: Verify Setup

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

## Step 12: Install Self-Hosted GitHub Actions Runner

For production environments behind firewalls or accessible through jumpserver/bastion hosts, a self-hosted runner allows direct deployment without SSH tunneling.

### Why Self-Hosted?

- Works regardless of network topology (behind firewall, jumpserver access, etc.)
- No need to expose SSH to the internet
- No need to manage SSH keys for deployment
- Faster deployments (local execution)

### Installing the Runner

1. Go to your GitHub repository
2. Navigate to **Settings** → **Actions** → **Runners**
3. Click **New self-hosted runner**
4. Choose **Linux** as the OS
5. Run the provided commands on your production server:

```bash
# Switch to deployment user
sudo su - sharpie-deploy

# Create a directory for the runner
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download the runner package (use the URL from GitHub UI)
curl -o actions-runner-linux-x64-<version>.tar.gz <download-url-from-github>

# Extract the package
tar xzf ./actions-runner-linux-x64-<version>.tar.gz

# Configure the runner (use the token from GitHub UI)
./config.sh --url https://github.com/<your-org>/<your-repo> --token <token-from-github>
```

6. When prompted:
   - **Runner group**: Press Enter for default
   - **Runner name**: `sharpie-production` (or your preferred name)
   - **Runner labels**: Press Enter to accept `self-hosted` (this must match the workflow)
   - **Work folder**: Press Enter for default `_work`

7. Exit from sharpie-deploy user:
```bash
exit
```

8. Install as a systemd service (run as root/sudo):
```bash
cd /home/sharpie-deploy/actions-runner
sudo ./svc.sh install sharpie-deploy
sudo ./svc.sh start
```

### Verifying the Runner

1. Go to GitHub repository → **Settings** → **Actions** → **Runners**
2. You should see your runner with a green "Online" status
3. Test by triggering the workflow manually:
   - Go to **Actions** tab
   - Select **Deploy to Production** workflow
   - Click **Run workflow**

### Runner Security

- The runner has access to the production server
- Using a dedicated user (sharpie-deploy) limits the scope
- Never run the runner as root
- The runner only has access to repositories you explicitly grant
- Consider using runner groups for additional isolation

### Runner Maintenance

- The runner will auto-update for minor versions
- For major updates, repeat the installation process
- To check runner status:
  ```bash
  sudo ./svc.sh status
  ```
- To view runner logs:
  ```bash
  journalctl -u actions.runner.SHARPIE.sharpie-production -f
  ```
- To stop/update/restart the runner:
  ```bash
  sudo ./svc.sh stop
  sudo ./svc.sh start
  sudo ./svc.sh restart
  ```

## Next Steps

After server setup is complete:
1. Install and verify the self-hosted runner (Step 12)
2. Push to `main` branch to trigger first deployment
3. Monitor the deployment in GitHub Actions:
   - Go to your repository on GitHub
   - Click the **Actions** tab
   - Click on the running workflow to view logs in real-time
4. Verify the application is working correctly:
   - Visit `https://your-domain.com` in a browser
   - Check service status on server: `supervisorctl status`