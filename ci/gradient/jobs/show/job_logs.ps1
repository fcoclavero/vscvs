Write-Host "Retrieving logs for job ID: " $args[0]
gradient jobs logs --jobId $args[0]
