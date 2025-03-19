$outputFile = "system_info.txt"


"`n=== Operating System Type and Version ===" | Out-File -Append $outputFile
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture | Format-Table -AutoSize | Out-File -Append $outputFile

"`n=== Number of Users Logged In ===" | Out-File -Append $outputFile
Get-CimInstance Win32_ComputerSystem | Select-Object UserName | Format-Table -AutoSize | Out-File -Append $outputFile

"=== Clock Speed, Processor Type, Core Count (Multithreading) ===" | Out-File -Append $outputFile
Get-CimInstance Win32_Processor | Select-Object Name, MaxClockSpeed, NumberOfCores, NumberOfLogicalProcessors | Format-Table -AutoSize | Out-File -Append $outputFile

"`n=== Cache Sizes (L1, L2, L3) ===" | Out-File -Append $outputFile
Get-CimInstance Win32_CacheMemory | Format-Table CacheType, InstalledSize -AutoSize | Out-File -Append $outputFile

"`n=== GPU Information (NVIDIA) ===" | Out-File -Append $outputFile
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used --format=csv | Out-File -Append $outputFile
} else {
    "nvidia-smi not found. Ensure NVIDIA drivers and CUDA are installed." | Out-File -Append $outputFile
}

"`n=== Top 10 CPU-consuming Processes ===" | Out-File -Append $outputFile
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 | Format-Table -AutoSize | Out-File -Append $outputFile

"`n=== Top 10 Memory-consuming Processes ===" | Out-File -Append $outputFile
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10 | Format-Table -AutoSize | Out-File -Append $outputFile

"`n=== Available Memory ===" | Out-File -Append $outputFile
Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory | Format-Table -AutoSize | Out-File -Append $outputFile


"`n=== GCC Version ===" | Out-File -Append $outputFile
if (Get-Command gcc -ErrorAction SilentlyContinue) {
    gcc --version 2>&1 | Out-File -Append $outputFile
} else {
    "GCC not found. Ensure it's installed and in PATH." | Out-File -Append $outputFile
}

"`n=== CUDA Version (nvcc) ===" | Out-File -Append $outputFile
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    nvcc --version 2>&1 | Out-File -Append $outputFile
} else {
    "nvcc not found. Ensure CUDA Toolkit is installed and in PATH." | Out-File -Append $outputFile
}


"System information collected in $outputFile"
