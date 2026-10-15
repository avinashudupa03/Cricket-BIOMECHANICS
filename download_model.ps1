$wc = New-Object System.Net.WebClient
try {
    $wc.DownloadFile('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task', 'E:\Cricket-BIOMECHANICS-main1\models\pose_landmarker_heavy.task')
    Write-Host 'Download complete'
} catch {
    Write-Host ('Error: ' + $_.Exception.Message)
}