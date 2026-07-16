# install.ps1 — ARDYローカルエンジンのセットアップスクリプト (Windows)
#
# 実行例 (PowerShell):
#   powershell -ExecutionPolicy Bypass -File tools\ardy-engine\install.ps1
#
# やること:
#   1. Python 3.10+ / git / (NVIDIA GPUなら) CUDA対応PyTorch の確認とvenv作成
#   2. C++ビルドツール (MinGW) をwingetで導入し、ARDY本体をビルドインストール
#   3. ARDYモデル重みのダウンロード
#   4. テキストエンコーダ (Llama-3-8B + LLM2Vec mntp) の構築
#      - 既定: NousResearchミラー (Meta Llama 3 Community License の正規再配布) を使用
#   5. アプリが自動起動に使う設定ファイル (ardy-engine.json) の書き出し
#
# 必要ディスク: 約35GB / 必要RAM: 16GB以上
param(
    [string]$EngineRoot = "$env:LOCALAPPDATA\text-to-vrma\ardy-engine"
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== ARDY Local Engine Setup ===" -ForegroundColor Cyan
Write-Host "install root: $EngineRoot"

# --- 1. 前提確認 ---
$py = $null
foreach ($cand in @('py -3.12', 'py -3.11', 'py -3.10', 'python')) {
    try {
        $v = Invoke-Expression "$cand --version" 2>$null
        if ($v -match 'Python 3\.(1[0-9])') { $py = $cand; break }
    } catch {}
}
if (-not $py) { throw "Python 3.10以上が見つかりません。https://www.python.org/ からインストールしてください。" }
Write-Host "Python: $py"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "gitが見つかりません。" }

$hasNvidia = $false
try { $null = nvidia-smi 2>$null; $hasNvidia = ($LASTEXITCODE -eq 0) } catch {}
Write-Host "NVIDIA GPU: $(if ($hasNvidia) {'あり (CUDA版PyTorchを使用)'} else {'なし (CPU版PyTorch、生成に数十秒/回)'})"

# --- 2. MinGW (C++ビルドツール) ---
$mingwPkg = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe"
$gxx = Get-ChildItem -Path $mingwPkg -Filter 'g++.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $gxx) {
    Write-Host "MinGW GCC をインストールします (winget)..."
    winget install BrechtSanders.WinLibs.POSIX.UCRT --accept-source-agreements --accept-package-agreements --disable-interactivity
    $gxx = Get-ChildItem -Path $mingwPkg -Filter 'g++.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (-not $gxx) { throw "MinGWのインストールに失敗しました。" }
$mingwBin = $gxx.DirectoryName
Write-Host "MinGW: $mingwBin"

# --- 3. venv + ARDY本体 ---
New-Item -ItemType Directory -Force $EngineRoot | Out-Null
$venvPy = Join-Path $EngineRoot 'venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    Invoke-Expression "$py -m venv `"$EngineRoot\venv`""
}
& $venvPy -m pip install --upgrade pip --quiet

Write-Host "PyTorch をインストール中... (数GB)"
if ($hasNvidia) {
    & $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu128
} else {
    & $venvPy -m pip install torch
}

$ardyRepo = Join-Path $EngineRoot 'ardy'
if (-not (Test-Path "$ardyRepo\setup.py")) {
    git clone --depth 1 https://github.com/nv-tlabs/ardy.git $ardyRepo
}
Write-Host "ARDY をビルド中... (C++拡張のコンパイルを含む)"
$env:PATH = "$mingwBin;$env:PATH"
& $venvPy -m pip install cmake sentencepiece --quiet
Push-Location $ardyRepo
& $venvPy -m pip install -e .
Pop-Location

# --- 4. モデル重み ---
Write-Host "ARDYモデル重みをダウンロード中..."
& $venvPy -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/ARDY-Core-RP-20FPS-Horizon40')"

$mergedBase = Join-Path $EngineRoot 'llm2vec-base-merged'
if (-not (Test-Path "$mergedBase\model.safetensors")) {
    Write-Host "テキストエンコーダを構築中... (約16GBのダウンロード + マージ。時間がかかります)"
    & $venvPy (Join-Path $ScriptDir 'build_text_encoder.py') --out $mergedBase
}

# --- 5. アプリ用設定ファイル ---
$config = @{
    pythonExe         = $venvPy
    mergedBase        = $mergedBase
    port              = 2337
    textEncoderDevice = 'cpu'
} | ConvertTo-Json
# パッケージ版とdev版 (electron .) の両方のuserDataに書く
foreach ($dir in @("$env:APPDATA\text-to-vrma", "$env:APPDATA\Electron")) {
    New-Item -ItemType Directory -Force $dir | Out-Null
    $config | Out-File -Encoding utf8 (Join-Path $dir 'ardy-engine.json')
}

Write-Host ""
Write-Host "=== セットアップ完了 ===" -ForegroundColor Green
Write-Host "アプリの「ARDYローカルエンジン」モードから「エンジンを起動」で使えます。"
Write-Host "手動起動: `"$venvPy`" `"$ScriptDir\server.py`" --merged-base `"$mergedBase`""
Write-Host ""
Write-Host "本エンジンは Meta Llama 3 を利用しています (Built with Meta Llama 3)。"
Write-Host "ライセンス: ARDY=NVIDIA Open Model / Llama-3-8B=Meta Llama 3 Community License / FuguMT=CC BY-SA 4.0"
