# Deployment Guide: Hosting Multiple Websites Without Conflicts

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Why Old Sites Break](#why-old-sites-break)
3. [Platform-Specific Fixes](#platform-specific-fixes)
4. [SPA Routing Configuration](#spa-routing-configuration)
5. [Safe Deployment Checklist](#safe-deployment-checklist)
6. [Recommended Architecture for cellcode.org](#recommended-architecture-for-cellcodeorg)
7. [Hosting Recommendations](#hosting-recommendations)
8. [Testing and Rollback Strategy](#testing-and-rollback-strategy)

---

## Understanding the Problem

You have two websites to host:
1. **Journal Site** - Static single-page site (journal.cellcode.org)
2. **E-MAS ML App** - Streamlit application (app.cellcode.org)

**The Core Issue**: When you deploy a new site, it overwrites or conflicts with the existing site, causing the old one to stop working.

---

## Why Old Sites Break

### Root Causes

| Cause | Explanation | Impact |
|-------|-------------|--------|
| **Same Project/Site ID** | Deploying to the same project overwrites previous deployment | Old site completely replaced |
| **Same Output Directory** | Build output goes to same folder, overwriting files | Mixed/old files remain, causing 404s |
| **DNS/Domain Changes** | Domain pointed to new deployment | Old site becomes unreachable |
| **SPA Routing Issues** | Deep links break without proper redirects | 404 errors on refresh |
| **CDN Caching** | Old assets cached, new assets not propagated | Users see broken/outdated site |
| **Shared Resources** | Both sites use same storage/database | Data corruption or conflicts |

### The Golden Rule

> **Each website must have its own isolated deployment target - never share project IDs, output folders, or domains between different sites.**

---

## Platform-Specific Fixes

### 1. GitHub Pages

**Why sites break on GitHub Pages:**
- Pushing to the same repository overwrites the previous site
- Using the same `gh-pages` branch replaces content

**Solution: Separate Repositories**

```bash
# Repository 1: Journal Site
# Repo: github.com/yourusername/journal-site
git clone https://github.com/yourusername/journal-site.git
cd journal-site
# ... build ...
git add dist/
git commit -m "Deploy journal site"
git push origin main

# Repository 2: E-MAS App (separate repo!)
# Repo: github.com/yourusername/emas-app
git clone https://github.com/yourusername/emas-app.git
cd emas-app
# ... build ...
git add .
git commit -m "Deploy E-MAS app"
git push origin main
```

**Settings for each repository:**

1. Go to **Settings → Pages**
2. Source: Deploy from a branch
3. Branch: `gh-pages` /root (or `main` /root)
4. **Custom domain**: 
   - Repo 1: `journal.cellcode.org`
   - Repo 2: `app.cellcode.org`

**DNS Configuration:**
```
# In your DNS provider (Cloudflare, Namecheap, etc.)
CNAME  journal  yourusername.github.io
CNAME  app      yourusername.github.io
```

---

### 2. Netlify

**Why sites break on Netlify:**
- Deploying to the same site ID overwrites
- Team members may accidentally deploy to wrong site

**Solution: Separate Sites with Unique IDs**

**Step 1: Create Two Separate Sites**

```bash
# Site 1: Journal
netlify sites:create --name journal-cellcode
# Output: Site ID: 1234abcd-1234-1234-1234-123456789abc

# Site 2: E-MAS App  
netlify sites:create --name emas-app-cellcode
# Output: Site ID: 5678efgh-5678-5678-5678-567856785678
```

**Step 2: Configure netlify.toml for each site**

For **Journal Site** (`netlify.toml`):
```toml
[build]
  publish = "dist"
  command = "npm run build"

[build.environment]
  NODE_VERSION = "20"

# SPA Routing - CRITICAL for React/Vue/Angular
[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-XSS-Protection = "1; mode=block"

# Site-specific settings
[site]
  id = "1234abcd-1234-1234-1234-123456789abc"
```

For **E-MAS App** (`netlify.toml`):
```toml
[build]
  publish = "."
  command = "echo 'Streamlit app - no build needed'"

# IMPORTANT: Streamlit apps need different handling
# Option A: Deploy as static (limited)
# Option B: Use Netlify Functions (advanced)
# Option C: Use dedicated Streamlit hosting (recommended)

[site]
  id = "5678efgh-5678-5678-5678-567856785678"
```

**Step 3: Deploy with explicit site IDs**

```bash
# Deploy Journal Site
netlify deploy --prod --site=1234abcd-1234-1234-1234-123456789abc --dir=dist

# Deploy E-MAS App (if using static export)
netlify deploy --prod --site=5678efgh-5678-5678-5678-567856785678 --dir=.
```

**Alternative: Use _redirects file (simpler)**

Create `dist/_redirects` for Journal Site:
```
# SPA Routing - send all routes to index.html
/*    /index.html   200

# Optional: Custom redirects
/old-page    /new-page    301
```

**Domain Configuration in Netlify:**
1. Go to Site settings → Domain management
2. Add custom domain: `journal.cellcode.org`
3. Configure DNS as instructed
4. Repeat for `app.cellcode.org` on the other site

---

### 3. Vercel

**Why sites break on Vercel:**
- `vercel --prod` deploys to linked project
- Same project name causes conflicts

**Solution: Separate Projects with vercel.json**

**Step 1: Create separate projects**

```bash
# Project 1: Journal Site
mkdir journal-site && cd journal-site
vercel projects add journal-cellcode
vercel link  # Select journal-cellcode

# Project 2: E-MAS App
mkdir emas-app && cd emas-app
vercel projects add emas-app-cellcode
vercel link  # Select emas-app-cellcode
```

**Step 2: Configure vercel.json for each project**

For **Journal Site** (`vercel.json`):
```json
{
  "version": 2,
  "name": "journal-cellcode",
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "dist"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        }
      ]
    }
  ],
  "github": {
    "enabled": false
  }
}
```

For **E-MAS App** (`vercel.json`):
```json
{
  "version": 2,
  "name": "emas-app-cellcode",
  "builds": [
    {
      "src": "api/*.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/api/index.py"
    }
  ]
}
```

**Step 3: Deploy with explicit project**

```bash
# Deploy Journal Site
cd journal-site
vercel --prod --confirm

# Deploy E-MAS App
cd emas-app
vercel --prod --confirm
```

**Domain Configuration:**
1. Go to Vercel Dashboard → Select project
2. Settings → Domains
3. Add `journal.cellcode.org`
4. Add DNS records as shown
5. Repeat for `app.cellcode.org`

---

### 4. Cloudflare Pages

**Why sites break on Cloudflare Pages:**
- Same project name overwrites
- Build commands affect all deployments

**Solution: Separate Projects with _redirects**

**Step 1: Create two separate projects**

In Cloudflare Dashboard:
1. Pages → Create a project
2. Project name: `journal-cellcode-org`
3. Repeat: Project name: `emas-app-cellcode-org`

**Step 2: Configure _redirects file**

For **Journal Site** (`dist/_redirects`):
```
# SPA Routing
/*    /index.html   200
```

For **E-MAS App** (`_redirects`):
```
# If using static export
/*    /index.html   200
```

**Step 3: Build configuration**

Journal Site:
- Build command: `npm run build`
- Build output: `dist`

E-MAS App (static export):
- Build command: `streamlit-static-export` (or custom script)
- Build output: `static`

**Step 4: Custom domains**

1. Go to Pages → Select project → Custom domains
2. Add `journal.cellcode.org`
3. Cloudflare will automatically configure DNS
4. Repeat for `app.cellcode.org`

---

### 5. University Server / cPanel (Folder-Based Hosting)

**Why sites break on shared hosting:**
- Uploading to `public_html` overwrites everything
- No separation between projects

**Solution: Separate Subdirectories + Subdomains**

**Option A: Separate Subdomains (Recommended)**

```
/public_html/                    (main domain - redirect or landing page)
/journal/                        (journal.cellcode.org document root)
    index.html
    assets/
    ...
/app/                            (app.cellcode.org document root)
    index.html
    ...
```

**cPanel Configuration:**

1. **Create Subdomains:**
   - Subdomains → Create
   - Subdomain: `journal`
   - Domain: `cellcode.org`
   - Document Root: `public_html/journal`
   
   - Subdomain: `app`
   - Domain: `cellcode.org`
   - Document Root: `public_html/app`

2. **Upload files:**
```bash
# Journal site
scp -r dist/* user@server:~/public_html/journal/

# E-MAS App (if static)
scp -r app/* user@server:~/public_html/app/
```

3. **SPA Routing (.htaccess for Apache):**

Create `/public_html/journal/.htaccess`:
```apache
<IfModule mod_rewrite.c>
  RewriteEngine On
  RewriteBase /
  
  # Don't rewrite files or directories
  RewriteCond %{REQUEST_FILENAME} -f [OR]
  RewriteCond %{REQUEST_FILENAME} -d
  RewriteRule ^ - [L]
  
  # Rewrite everything else to index.html
  RewriteRule ^ index.html [L]
</IfModule>

# Enable gzip compression
<IfModule mod_deflate.c>
  AddOutputFilterByType DEFLATE text/html text/css application/javascript
</IfModule>

# Cache static assets
<IfModule mod_expires.c>
  ExpiresActive On
  ExpiresByType image/jpeg "access plus 1 year"
  ExpiresByType text/css "access plus 1 month"
</IfModule>
```

Create `/public_html/app/.htaccess` (same content):
```apache
<IfModule mod_rewrite.c>
  RewriteEngine On
  RewriteBase /
  RewriteCond %{REQUEST_FILENAME} -f [OR]
  RewriteCond %{REQUEST_FILENAME} -d
  RewriteRule ^ - [L]
  RewriteRule ^ index.html [L]
</IfModule>
```

**Option B: Reverse Proxy (For Streamlit on same server)**

If running Streamlit on the same server:

```apache
# In /public_html/app/.htaccess or Apache config
RewriteEngine On
RewriteRule ^(.*)$ http://localhost:8501/$1 [P,L]

# Or using ProxyPass in Apache config
ProxyPass / http://localhost:8501/
ProxyPassReverse / http://localhost:8501/
```

---

## SPA Routing Configuration

### Why SPA Routing is Critical

Single Page Applications (React, Vue, Angular) handle routing client-side. When a user:
1. Navigates to `/dashboard` → Works (client-side routing)
2. Refreshes the page → 404 error (server doesn't know about `/dashboard`)

### Platform-Specific Solutions

| Platform | Config File | Rule |
|----------|-------------|------|
| **Netlify** | `_redirects` | `/* /index.html 200` |
| **Netlify** | `netlify.toml` | `[[redirects]] from="/*" to="/index.html" status=200` |
| **Vercel** | `vercel.json` | `"routes": [{"src":"/(.*)","dest":"/index.html"}]` |
| **Cloudflare** | `_redirects` | `/* /index.html 200` |
| **Apache** | `.htaccess` | `RewriteRule ^ index.html [L]` |
| **Nginx** | `nginx.conf` | `try_files $uri $uri/ /index.html;` |

### Complete _redirects File (Netlify/Cloudflare)

```
# SPA Routing - MUST be first
/*    /index.html   200

# Optional: API proxy (if needed)
/api/*  https://api.example.com/:splat  200

# Optional: Old page redirects
/old-page    /new-page    301
/another-old  /new-location  301

# Optional: Custom 404
/404    /404.html    200
```

### Complete vercel.json (Vercel)

```json
{
  "version": 2,
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/$1"
    },
    {
      "handle": "filesystem"
    },
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

### Complete .htaccess (Apache/cPanel)

```apache
# Enable rewrite engine
<IfModule mod_rewrite.c>
  RewriteEngine On
  RewriteBase /
  
  # Handle trailing slashes
  RewriteRule ^(.*)/$ /$1 [L,R=301]
  
  # Don't rewrite existing files/directories
  RewriteCond %{REQUEST_FILENAME} -f [OR]
  RewriteCond %{REQUEST_FILENAME} -d
  RewriteRule ^ - [L]
  
  # Rewrite all to index.html
  RewriteRule ^ index.html [L]
</IfModule>

# Security headers
<IfModule mod_headers.c>
  Header always set X-Frame-Options "DENY"
  Header always set X-Content-Type-Options "nosniff"
  Header always set X-XSS-Protection "1; mode=block"
  Header always set Referrer-Policy "strict-origin-when-cross-origin"
</IfModule>

# Compression
<IfModule mod_deflate.c>
  AddOutputFilterByType DEFLATE text/plain text/html text/css
  AddOutputFilterByType DEFLATE application/javascript application/json
  AddOutputFilterByType DEFLATE image/svg+xml
</IfModule>

# Caching
<IfModule mod_expires.c>
  ExpiresActive On
  ExpiresByType image/* "access plus 1 year"
  ExpiresByType text/css "access plus 1 month"
  ExpiresByType application/javascript "access plus 1 month"
</IfModule>
```

---

## Safe Deployment Checklist

Use this checklist EVERY time you deploy:

### Pre-Deployment

- [ ] **Verify unique site/project name**
  - Journal: `journal-cellcode` 
  - App: `emas-app-cellcode`
  - NEVER use the same name

- [ ] **Check domain/subdomain**
  - Journal: `journal.cellcode.org`
  - App: `app.cellcode.org`
  - Verify DNS records are correct

- [ ] **Confirm build output directory**
  - Journal: `dist/` (Vite/React)
  - App: `static/` or root
  - Different folders for each project

- [ ] **SPA routing configured**
  - `_redirects` or `vercel.json` or `.htaccess` in place
  - Tested locally with `npm run preview`

### Deployment

- [ ] **Deploy to correct project**
  ```bash
  # Double-check before running!
  vercel --prod  # Confirm project name
  netlify deploy --prod --site=CORRECT_SITE_ID
  ```

- [ ] **Monitor build logs**
  - Check for errors
  - Verify build succeeds

### Post-Deployment

- [ ] **Test the new site**
  - Homepage loads: `https://new.cellcode.org`
  - Deep links work: `https://new.cellcode.org/dashboard`
  - Refresh works on all pages

- [ ] **Verify old site still works**
  - Open `https://old.cellcode.org` in incognito
  - Test multiple pages
  - Clear cache and test again (Ctrl+Shift+R)

- [ ] **Check both sites simultaneously**
  ```bash
  # Test both in parallel
  curl -I https://journal.cellcode.org
  curl -I https://app.cellcode.org
  ```

- [ ] **Document deployment**
  - Note deployment time
  - Record any issues
  - Save rollback procedure

---

## Recommended Architecture for cellcode.org

### Option 1: Separate Subdomains (Recommended)

```
cellcode.org                    → Landing page or redirect
├── journal.cellcode.org        → Static site (React/Vite)
│   └── Hosted on: Netlify/Vercel/Cloudflare Pages
│   └── Build: npm run build → dist/
│   └── Config: _redirects for SPA routing
│
└── app.cellcode.org            → Streamlit ML App
    └── Hosted on: Streamlit Community Cloud / Render / Railway
    └── Build: Docker or native Python
    └── Config: Separate deployment pipeline
```

**DNS Configuration:**
```
Type    Name        Value                           TTL
A       @           192.0.2.1 (landing page IP)     Auto
CNAME   journal     cname.netlify.com               Auto
CNAME   app         share.streamlit.io              Auto
```

### Option 2: Subpaths with Reverse Proxy

```
cellcode.org                    → Main server (Nginx/Apache)
├── /journal/*                  → Proxy to journal static files
│   └── location /journal/ {
│       alias /var/www/journal/;
│       try_files $uri $uri/ /journal/index.html;
│   }
│
└── /app/*                      → Proxy to Streamlit
    └── location /app/ {
        proxy_pass http://localhost:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
```

**Recommendation**: Use **Option 1 (Separate Subdomains)** - it's cleaner, easier to manage, and each service can use its optimal hosting platform.

---

## Hosting Recommendations

### Journal Site (Static)

| Platform | Best For | Free Tier | Custom Domain |
|----------|----------|-----------|---------------|
| **Vercel** | React/Next.js | ✅ Unlimited | ✅ |
| **Netlify** | General static | ✅ 100GB/mo | ✅ |
| **Cloudflare Pages** | Speed/CDN | ✅ Unlimited | ✅ |
| **GitHub Pages** | Simple sites | ✅ 1GB | ✅ |

**Recommended**: Cloudflare Pages or Vercel

### E-MAS Streamlit App

| Platform | Best For | Free Tier | Docker Support |
|----------|----------|-----------|----------------|
| **Streamlit Community Cloud** | Streamlit apps | ✅ 1GB RAM | ❌ |
| **Render** | Full stack | ✅ 512MB RAM | ✅ |
| **Railway** | Easy deploy | ✅ $5 credit | ✅ |
| **Fly.io** | Global edge | ✅ $5 credit | ✅ |
| **Hugging Face Spaces** | ML demos | ✅ 16GB RAM | ✅ |

**Recommended**: 
- **Streamlit Community Cloud** (simplest, free)
- **Render** (Docker support, persistent)

### Streamlit Community Cloud Deployment

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Select `app.py` as main file
5. Deploy
6. Add custom domain: `app.cellcode.org`

### Render Deployment

Create `render.yaml`:
```yaml
services:
  - type: web
    name: emas-skin-lesion-classifier
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

---

## Testing and Rollback Strategy

### Pre-Deployment Testing

```bash
# 1. Build locally
npm run build

# 2. Test build output
npx serve dist

# 3. Test SPA routing
curl http://localhost:3000/dashboard  # Should return index.html

# 4. Test on staging (if available)
vercel --target=preview
```

### Post-Deployment Testing

```bash
#!/bin/bash
# test-deployment.sh

SITES=(
  "https://journal.cellcode.org"
  "https://journal.cellcode.org/about"
  "https://app.cellcode.org"
)

for site in "${SITES[@]}"; do
  echo "Testing: $site"
  status=$(curl -s -o /dev/null -w "%{http_code}" "$site")
  if [ "$status" = "200" ]; then
    echo "✅ OK (200)"
  else
    echo "❌ FAILED ($status)"
  fi
done
```

### Rollback Procedures

**Vercel Rollback:**
```bash
# List deployments
vercel ls

# Rollback to previous
vercel --prod --confirm DEPLOYMENT_URL
```

**Netlify Rollback:**
```bash
# List deploys
netlify deploys:list --site=SITE_ID

# Rollback
netlify deploys:restore DEPLOY_ID --site=SITE_ID --prod
```

**GitHub Pages Rollback:**
```bash
# Revert to previous commit
git revert HEAD
git push

# Or force push old commit
git push --force origin OLD_COMMIT:gh-pages
```

**General Rollback (Any Platform):**
1. Identify last working deployment
2. Redeploy that specific version
3. Update DNS if domain changed
4. Clear CDN cache

---

## Summary: Key Takeaways

1. **Never share project IDs** - Each site gets its own
2. **Never share build directories** - Separate output folders
3. **Always configure SPA routing** - `_redirects`, `vercel.json`, or `.htaccess`
4. **Use separate subdomains** - `journal.` and `app.`
5. **Test both sites after each deployment** - Old AND new
6. **Have a rollback plan** - Know how to revert quickly
7. **Document everything** - Keep deployment records

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT CHECKLIST                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Journal Site (journal.cellcode.org)                        │
│  ├── Platform: Vercel/Netlify/Cloudflare                    │
│  ├── Build: npm run build → dist/                           │
│  ├── SPA Config: _redirects or vercel.json                  │
│  └── Deploy: vercel --prod / netlify deploy --prod          │
│                                                              │
│  E-MAS App (app.cellcode.org)                               │
│  ├── Platform: Streamlit Cloud / Render                     │
│  ├── Build: streamlit run app.py                            │
│  ├── Config: requirements.txt + Dockerfile                  │
│  └── Deploy: Git push → Auto-deploy                         │
│                                                              │
│  After Deployment:                                          │
│  ├── ☐ Test journal.cellcode.org                            │
│  ├── ☐ Test app.cellcode.org                                │
│  ├── ☐ Test deep links on both                              │
│  └── ☐ Clear browser cache and retest                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

**Need Help?** 
- Check platform documentation: [Vercel](https://vercel.com/docs), [Netlify](https://docs.netlify.com), [Cloudflare](https://developers.cloudflare.com/pages)
- Streamlit deployment: [docs.streamlit.io](https://docs.streamlit.io/streamlit-community-cloud)
- Open an issue in this repository
