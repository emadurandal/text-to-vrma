# Text to VRMA — テキストからVRMアニメーション生成

テキストを入力すると、ChatGPT (OpenAI API) がキーフレームを設計し、
**VRMA (VRM Animation / `.vrma`)** ファイルをブラウザ内で生成して、
その場で VRM キャラクターを動かす Web アプリです。
生成した `.vrma` はファイルとして保存でき、VRMA 対応アプリでそのまま利用できます。

例:「忍者のように構えてから手裏剣を投げる」「手を振ってからお辞儀する」「大きくジャンプを3回」

## 必要なもの

- Node.js 20+
- OpenAI API キー ([platform.openai.com](https://platform.openai.com/) で取得)

VRM モデルはサンプル (`public/models/SampleBot.vrm`、本リポジトリ生成のローポリ人間) を
同梱しているのでそのまま試せます。手持ちの `.vrm` への差し替えも可能です。

`public/models/AvatarSample_A.vrm` に VRoid 公式サンプルモデルを置くと、
起動時にそちらが優先して読み込まれます
([VRoid Hub の AvatarSample](https://hub.vroid.com/characters/2843975675147313744/models/5644550979324015604)
からダウンロード。無償での再配布は許可されていますが、有償再配布と CC0 としての
配布は禁止されています — モデルは MIT ライセンスの対象外です)。

VRM **0.x / 1.0 の両形式に対応**しています (three-vrm が自動判別し、向きも正規化)。

## セットアップ & 起動

```sh
git clone <このリポジトリ>
cd text-to-motion
npm install
npm run dev
# → http://localhost:5173 をブラウザで開く
```

## 使い方

1. 起動するとサンプルモデル (SampleBot) が読み込まれます。
   「VRMファイルを開く」または 3D ビューへのドラッグ&ドロップで手持ちの VRM に差し替え可能
   - `public/models/Zundamon.vrm` など好きなモデルを置くと起動時に優先読み込みされます
2. OpenAI API キーを入力し、モデル (gpt-5.6 系) を選択
   - キーはブラウザの localStorage にのみ保存され、OpenAI 以外には送信されません
3. テキストを入力して「▶ モーション生成 & 再生」 (Ctrl+Enter でも可)
4. 「⬇ .vrma 保存」で生成アニメーションをファイルに書き出し

## アーキテクチャ

```text
テキスト ── ChatGPT API ──▶ モーション spec (ボーン別オイラー角キーフレーム JSON)
                                   │
                                   ▼
              vrmaBuilder.js ── glTF + VRMC_vrm_animation 拡張 → GLB (.vrma)
                                   │
                                   ▼
              viewer.js ── three.js + @pixiv/three-vrm-animation で VRM に再生
```

| ファイル | 役割 |
| --- | --- |
| `src/llm.js` | ChatGPT へのプロンプト (ボーン規約・出力形式) と spec 検証 |
| `src/vrmaBuilder.js` | モーション spec から VRMA (GLB) をバイナリ生成。VRM1 規約の T ポーズ骨格を埋め込み、`VRMC_vrm_animation` 拡張でヒューマノイドボーンをマッピング |
| `src/viewer.js` | three.js シーン / VRM ロード / VRMA 再生 |
| `src/idleMotion.js` | 待機モーション (呼吸) |
| `src/main.js` | UI 結線 |
| `scripts/generateSampleVRM.mjs` | 同梱サンプル VRM (SampleBot) の生成スクリプト |

## モーション spec フォーマット

ChatGPT が生成する中間表現です:

```json
{
  "name": "wave",
  "duration": 2.6,
  "loop": true,
  "tracks": {
    "rightUpperArm": [
      { "t": 0, "r": [0, 0, 70] },
      { "t": 0.4, "r": [0, 0, -45] }
    ]
  },
  "hips": [ { "t": 0, "p": [0, 0, 0] } ]
}
```

- `r` = T ポーズからのオイラー角 [X, Y, Z] (度) / `p` = 腰位置オフセット (m)
- 座標規約: モデルは +Z 正面 / +X が左手側 (VRM 1.0 準拠)

## ライセンス / 注意

- コード・同梱サンプルモデル (SampleBot.vrm): MIT License
- サードパーティの VRM モデルは各モデルの利用規約に従ってください
- 生成される `.vrma` の利用は各自の責任で行ってください
