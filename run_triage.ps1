# Start Ollama if not already running
$ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProcess) {
    Start-Process -FilePath "ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

# Wait until Ollama is ready
$maxWait = 30
$waited = 0
$ready = $false
while ($waited -lt $maxWait) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host "Ollama is ready."
            $ready = $true
            break
        }
    } catch {
        Write-Host "Waiting for Ollama... ($waited s)"
    }
    Start-Sleep -Seconds 2
    $waited += 2
}

if (-not $ready) {
    Write-Host "Ollama did not start in time. Aborting."
    exit 1
}

# Run triage
& ".venv\Scripts\python.exe" auto_triage_file.py --file keywords.txt --time 1d --max 30
