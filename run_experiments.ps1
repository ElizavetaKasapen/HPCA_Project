param (
    [string]$logFile = "experiment_results_standard.txt",
    [string]$exeFile = "main_with_logs.exe"
)

# Clear previous log file
if (Test-Path $logFile) { Remove-Item $logFile }

$totalTime = 0
$totalAccuracy = 0
$numExperiments = 10

for ($i = 1; $i -le $numExperiments; $i++) {
    Write-Output "Starting Experiment $i..." | Tee-Object -Append -FilePath $logFile

    # Run the C++ program and capture output
    $output = & $exeFile

    # Append the full output to the log file
    Add-Content $logFile $output

   $output -split "`r?`n" | ForEach-Object { Write-Output $_ }

    # Extract training time and accuracy
    $trainingTimeMatch = [regex]::Match($output, "Training time: ([\d\.]+) seconds")
    $accuracyMatch = [regex]::Match($output, "Accuracy: ([\d\.]+)%")

    if ($trainingTimeMatch.Success -and $accuracyMatch.Success) {
        $trainingTime = [double]$trainingTimeMatch.Groups[1].Value
        $accuracy = [double]$accuracyMatch.Groups[1].Value

        $totalTime += $trainingTime
        $totalAccuracy += $accuracy
    }
    Write-Output "---------------------------" | Tee-Object -Append -FilePath $logFile
}

# Compute and log averages
$avgTime = $totalTime / $numExperiments
$avgAccuracy = $totalAccuracy / $numExperiments

Write-Output "`nAverage Training Time: $avgTime seconds" | Tee-Object -Append -FilePath $logFile
Write-Output "Average Accuracy: $avgAccuracy%" | Tee-Object -Append -FilePath $logFile
Write-Output "---------------------------" | Tee-Object -Append -FilePath $logFile
Write-Output "All experiments completed! Results saved in $logFile"
