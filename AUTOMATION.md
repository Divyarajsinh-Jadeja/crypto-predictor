# Free Automation Setup Guide

This project uses **GitHub Actions** for 100% free automation. No credit card required!

## 🚀 What Gets Automated

1. **Model Training** - Every Sunday at 2 AM UTC
2. **Predictions to Google Chat** - Every 6 hours (4 times daily)
3. **Keep API Alive** - Pings your Render service every 10 minutes

## 📋 Setup Instructions

### Step 1: Push to GitHub

If you haven't already, push your code to GitHub:

```bash
git add .
git commit -m "Setup automation workflows"
git push origin main
```

### Step 2: Add Secrets to GitHub

Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets:

| Secret Name | Value | Example |
|-------------|-------|---------|
| `GOOGLE_CHAT_WEBHOOK_URL` | Your Google Chat webhook URL | `https://chat.googleapis.com/v1/spaces/...` |
| `API_BASE_URL` | Your Render API URL | `https://crypto-predictor-akpl.onrender.com` |

### Step 3: Enable GitHub Actions

1. Go to your repository on GitHub
2. Click on **Actions** tab
3. If prompted, click **"I understand my workflows, go ahead and enable them"**

### Step 4: Test Manual Trigger

1. Go to **Actions** tab
2. Select **"Send Crypto Predictions to Google Chat"**
3. Click **"Run workflow"** → **"Run workflow"**
4. Check your Google Chat in a few seconds!

## 📅 Automation Schedules

### Training Workflow (`train_models.yml`)
- **When**: Every Sunday at 2 AM UTC
- **What**: Trains all 20 cryptocurrency models with latest data
- **Duration**: ~30-60 minutes
- **Output**: Updates `models/` directory with new `.h5` and `.pkl` files

### Predictions Workflow (`send_predictions.yml`)
- **When**: 4 times daily at 12 AM, 6 AM, 12 PM, 6 PM UTC
- **What**: Fetches predictions from API and sends to Google Chat
- **Duration**: ~10-30 seconds
- **Output**: Formatted prediction report in your Google Chat

### Keep Alive Workflow (`keep_alive.yml`)
- **When**: Every 10 minutes
- **What**: Pings your Render API to prevent it from sleeping
- **Duration**: ~2 seconds
- **Output**: Keeps your free Render service awake 24/7

## 💰 Cost Analysis

| Service | Free Tier Limit | Usage |
|---------|----------------|-------|
| **GitHub Actions** | 2,000 minutes/month | ~200 minutes/month |
| **Render** | Free web service | 1 API instance |
| **Google Chat** | Unlimited webhooks | Free |
| **MongoDB Atlas** | 512 MB storage | Optional |

**Total Cost: $0/month** ✅

## 🔧 Customization

### Change Prediction Frequency

Edit `.github/workflows/send_predictions.yml`:

```yaml
schedule:
  # Every 3 hours instead of 6
  - cron: '0 */3 * * *'
  
  # Only twice daily (6 AM and 6 PM UTC)
  - cron: '0 6,18 * * *'
  
  # Every hour
  - cron: '0 * * * *'
```

### Change Training Frequency

Edit `.github/workflows/train_models.yml`:

```yaml
schedule:
  # Train daily instead of weekly
  - cron: '0 2 * * *'
  
  # Train twice a week (Sunday and Wednesday)
  - cron: '0 2 * * 0,3'
```

## 📊 Monitoring

### View Workflow Runs
1. Go to GitHub repository → **Actions** tab
2. Click on any workflow to see execution history
3. Green checkmark ✅ = Success
4. Red X ❌ = Failed (click to see logs)

### View Logs
1. Click on a workflow run
2. Click on the job name (e.g., "send-predictions")
3. Expand any step to see detailed logs

## 🐛 Troubleshooting

### Predictions aren't sending to Google Chat
- Check that `GOOGLE_CHAT_WEBHOOK_URL` secret is set correctly
- Verify the webhook URL is valid by testing locally
- Check workflow logs for error messages

### API is sleeping despite keep-alive
- Render's free tier may still sleep after 15 min of inactivity
- Consider upgrading to Render's paid tier ($7/month) for 24/7 uptime
- Or use a different hosting service like Railway, Fly.io, or Heroku

### Training workflow is failing
- Check GitHub Actions logs for specific error
- Ensure `requirements.txt` includes all dependencies
- Training can timeout if it takes >2 hours (adjust `timeout-minutes`)

## 🎯 Next Steps

1. ✅ Push code to GitHub
2. ✅ Add secrets to repository settings
3. ✅ Enable GitHub Actions
4. ✅ Test manual workflow trigger
5. ✅ Wait for first automated run
6. ✅ Check Google Chat for predictions!

Enjoy your fully automated crypto prediction system! 🚀
