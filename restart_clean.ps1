# Clear Streamlit cache and restart
Write-Host "Stopping Streamlit..." -ForegroundColor Yellow
taskkill /F /IM streamlit.exe 2>$null

Write-Host "Clearing cache..." -ForegroundColor Yellow
Start-Sleep -Seconds 1

# Clear Streamlit cache directory
if (Test-Path "$env:USERPROFILE\.streamlit\cache") {
    Remove-Item "$env:USERPROFILE\.streamlit\cache" -Recurse -Force
    Write-Host "Cache cleared" -ForegroundColor Green
}

Write-Host "Starting Streamlit..." -ForegroundColor Yellow
Start-Sleep -Seconds 1
streamlit run main.py
