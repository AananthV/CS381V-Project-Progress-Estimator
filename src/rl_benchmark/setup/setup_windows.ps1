# Windows setup for rl_benchmark. Run from the project root: ./src/rl_benchmark/setup/setup_windows.ps1
# Creates .venv-sim (Python 3.12, CUDA PyTorch 2.7.1), pinned RoboCasa + robosuite checkouts under
# data/simulation, and RoboCasa's kitchen assets. Needs Git, uv, an NVIDIA GPU and ~30 GB of disk.
$ErrorActionPreference = 'Stop'
function Invoke-Checked {
    param([string]$Executable, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}
$repositories = @(
    @('robocasa', 'https://github.com/robocasa/robocasa.git', '3443cdf3dd20f9ae8ed41adba442059b506891d0'),
    @('robosuite', 'https://github.com/ARISE-Initiative/robosuite.git', '5ce6643f3092639d08f7b0f90ed1c6a84f50552c')
)
foreach ($repository in $repositories) {
    $destination = "data/simulation/$($repository[0])"
    if (-not (Test-Path -LiteralPath "$destination/.git")) {
        Invoke-Checked git @('clone', $repository[1], $destination)
        Invoke-Checked git @('-C', $destination, 'checkout', $repository[2])
    }
    $current = (& git -C $destination rev-parse HEAD).Trim()
    if ($current -ne $repository[2]) { throw "$destination is at $current; expected $($repository[2])" }
}
if (-not (Test-Path -LiteralPath '.venv-sim/Scripts/python.exe')) { Invoke-Checked uv @('venv', '.venv-sim', '--python', '3.12') }
$py = '.venv-sim/Scripts/python.exe'
Invoke-Checked uv @('pip', 'install', '--python', $py, 'torch==2.7.1', 'torchvision==0.22.1', '--index-url', 'https://download.pytorch.org/whl/cu128')
Invoke-Checked uv @('pip', 'install', '--python', $py, '-r', 'src/rl_benchmark/setup/requirements.txt')
foreach ($repository in $repositories) {
    Invoke-Checked uv @('pip', 'install', '--python', $py, '--no-deps', '-e', "data/simulation/$($repository[0])")
}
Invoke-Checked $py @('src/rl_benchmark/setup/download_assets.py', '--checkout', 'data/simulation/robocasa')
Write-Host 'Done. Next: .venv-sim/Scripts/python.exe src/rl_benchmark/download_data.py --task add_sweetener'
